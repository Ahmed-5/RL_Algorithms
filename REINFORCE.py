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
    total_loss = 0
    episode_count = 0
    count = 0
    
    model.train()
    model.to(device)
    
    with tqdm(range(start_step, steps + start_step)) as pbar:
        for step in pbar:
            # Reset the environment at the beginning of each episode
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_loss = 0
            
            # Lists to store episode history
            states_buffer = []
            actions_buffer = []
            rewards_buffer = []
            
            # Collect a trajectory
            while not done:
                state_t = torch.tensor(state).float().to(device)
                with torch.no_grad():
                    policy = model(state_t)
                
                action = strategy.select_action(policy.detach().cpu().numpy(), sample=True, softmax=True)
                next_state, reward, done, out_of_bounds, _ = env.step(action)
                
                if out_of_bounds:
                    done = True
                
                # Store transition
                states_buffer.append(state)
                actions_buffer.append(action)
                rewards_buffer.append(reward)
                
                episode_reward += reward
                episode_steps += 1
                
                if tb_writer is not None:
                    tb_writer.add_scalar('Global/reward', reward, step * 1000 + episode_steps)
                
                state = next_state
                
                if done or out_of_bounds:
                    break
            
            # End of episode processing
            if len(rewards_buffer) > 0:
                # Convert lists to tensors
                states = torch.tensor(np.array(states_buffer), dtype=torch.float32).to(device)
                actions = torch.tensor(np.array(actions_buffer), dtype=torch.int64).to(device)
                rewards = torch.tensor(np.array(rewards_buffer), dtype=torch.float32).to(device)
                
                # Calculate returns
                returns = compute_cumulative_returns(rewards, gamma)
                
                # Get action probabilities
                logits = model(states)
                
                # Calculate loss
                loss = reinforce_loss(logits, actions, returns)
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 100)
                optimizer.step()
                
                if scheduler is not None:
                    if tb_writer is not None:
                        tb_writer.add_scalar("Global/lr", scheduler.get_last_lr()[-1], step)
                    scheduler.step(step)
                
                batch_loss = loss.item()
                count += 1
                total_loss += (batch_loss - total_loss) / count
                episode_loss = batch_loss
                
                if tb_writer is not None:
                    tb_writer.add_scalar("Global/loss", batch_loss, step)
                    tb_writer.add_scalar('Episode/reward', episode_reward, episode_count)
                    tb_writer.add_scalar('Episode/steps', episode_steps, episode_count)
                    tb_writer.add_scalar('Episode/loss', episode_loss, episode_count)
                
                episode_count += 1
                
                pbar.set_description(f"step: {(step+1):3d}/{steps+start_step}, ep_reward: {episode_reward:.0f}, ep_loss: {episode_loss:.2f}, avg_loss: {(total_loss):.2e}")
            
            # Save the model
            if (step+1) % save_every == 0 and save_path is not None:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode_count,
                    'step': step+1,
                    'loss': total_loss,
                }, save_path)