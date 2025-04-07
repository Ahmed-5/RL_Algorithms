import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class REINFORCE(nn.Module):
    def __init__(self, input_size, n_actions):
        super(REINFORCE, self).__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.unsqueeze = False

        if len(input_size) == 3 or len(input_size) == 2:
            self.unsqueeze = len(input_size) == 2
            channels = 1 if len(input_size) == 2 else input_size[0]
            h = input_size[-2]
            w = input_size[-1]
            
            # Fixed the calculation for output size after convolutions
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Flatten()
            )
            
            # Calculate the correct output size
            x = (h + 2*1 - 3) // 2 + 1  # First conv
            x = (x + 2*1 - 3) // 2 + 1  # Second conv
            
            y = (w + 2*1 - 3) // 2 + 1  # First conv
            y = (y + 2*1 - 3) // 2 + 1  # Second conv
            
            self.fc = nn.Sequential(
                nn.Linear(x * y * 64, 512),
                nn.GELU(),
                nn.Linear(512, n_actions)
            )
            
        elif len(input_size) == 1:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_size[0], 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU()
            )
            self.fc = nn.Linear(128, n_actions)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.unsqueeze:
            x = x.unsqueeze(1)
        features = self.feature_extractor(x)
        logits = self.fc(features)
        return self.softmax(logits)
    

def compute_cumulative_returns(rewards, gamma):
    """
    Compute discounted cumulative future rewards for each time step.
    R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    """
    returns = torch.zeros_like(rewards)
    future_return = 0
    
    # Compute returns from the end to the beginning
    for t in range(len(rewards) - 1, -1, -1):
        future_return = rewards[t] + gamma * future_return
        returns[t] = future_return
        
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def reinforce_loss(logits, actions, returns, entropy_coef=0.01):
    """Calculate policy gradient loss with entropy regularization"""
    log_probs = torch.log(logits + 1e-10)  # Add small epsilon to prevent log(0)
    entropy = -(logits * log_probs).sum(dim=1)
    
    # Get log probabilities of the taken actions
    actions = actions.unsqueeze(1)
    selected_log_probs = log_probs.gather(1, actions).squeeze(1)
    
    # Policy gradient loss
    pg_loss = -(selected_log_probs * returns).mean()
    
    # Entropy regularization loss
    entropy_loss = -entropy.mean()
    
    return pg_loss + entropy_coef * entropy_loss


def train_reinforce(
        model, 
        steps, 
        optimizer,  
        device, 
        env,
        replay_memory, 
        strategy,
        batch_size,
        gamma=0.99,
        start_step=0,
        save_path=None,
        save_every=10,
        tb_writer=None,
        scheduler=None
    ):
    done = True
    total_loss = 0
    episode_count = 0
    reward = 0
    er = episode_reward = 0
    es = episode_steps = 0
    el = episode_loss = 0
    count = 0

    model.train()
    model.to(device)
    
    with tqdm(range(start_step, steps + start_step)) as pbar:
        for step in pbar:
            if done:
                state, _ = env.reset()
            
            state_t = torch.tensor(state).float().to(device)
            with torch.no_grad():
                policy = model(state_t)
            
            action = strategy.select_action(policy.detach().cpu().numpy(), sample=True, softmax=True)
            next_state, reward, done, out_of_bounds, _ = env.step(action)
            
            if out_of_bounds:
                done = True
                
            if tb_writer is not None:
                tb_writer.add_scalar('Global/reward', reward, step+1)

            episode_reward += reward
            episode_steps += 1
            count += 1
            
            # Push the transition to replay memory
            replay_memory.push(state, action, reward, next_state, done)

            # Sample episodes from replay memory
            batch = replay_memory.sample_episodes(batch_size)
            
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                batch_loss = 0
                
                # Process each episode in the batch
                for i in range(len(states)):
                    episode_states = torch.tensor(states[i], dtype=torch.float32).to(device)
                    episode_actions = torch.tensor(actions[i], dtype=torch.int64).to(device)
                    episode_rewards = torch.tensor(rewards[i], dtype=torch.float32).to(device)
                    
                    # Calculate returns for this episode
                    episode_returns = compute_cumulative_returns(episode_rewards, gamma)
                    
                    # Get action probabilities for this episode
                    episode_logits = model(episode_states)
                    
                    # Calculate loss for this episode
                    episode_loss = reinforce_loss(episode_logits, episode_actions, episode_returns)
                    batch_loss += episode_loss
                
                # Average loss across all episodes in the batch
                batch_loss = batch_loss / len(states)
                
                # Update model
                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 100)
                optimizer.step()
                
                if scheduler is not None:
                    if tb_writer is not None:
                        tb_writer.add_scalar("Global/lr", scheduler.get_last_lr()[-1], step+1)
                    scheduler.step(step)
                
                loss_value = batch_loss.item()
                total_loss += (loss_value - total_loss) / count
                episode_loss += (loss_value - episode_loss) / episode_steps
                
                if tb_writer is not None:
                    tb_writer.add_scalar("Global/loss", loss_value, step+1)
            
            # Handle episode completion
            if done:
                if tb_writer is not None:
                    tb_writer.add_scalar('Episode/reward', episode_reward, episode_count+1)
                    tb_writer.add_scalar('Episode/steps', episode_steps, episode_count+1)
                    tb_writer.add_scalar('Episode/loss', episode_loss, episode_count+1)

                episode_count += 1
                er = episode_reward
                es = episode_steps
                el = episode_loss
                episode_reward = 0
                episode_steps = 0
                episode_loss = 0
            else:
                state = next_state
            
            pbar.set_description(f"step: {(step+1):3d}/{steps+start_step}, ep_reward: {er:.0f}, ep_loss: {el:.2f}, avg_loss: {total_loss:.2e}")

            # Save model checkpoint
            if (step+1) % save_every == 0 and save_path is not None:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode_count,
                    'step': step+1,
                    'loss': total_loss,
                }, save_path)