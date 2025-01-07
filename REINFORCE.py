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

        self.layers = []
        if len(input_size) == 3 or len(input_size) == 2:
            self.unsqueeze = len(input_size) == 2
            channels = 1 if len(input_size) == 2 else input_size[0]
            h = input_size[-2]
            w = input_size[-1]
            self.layers.append(nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.GELU())
            self.layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.GELU())
            self.layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.GELU())
            x, y = ((h + 1 // 2) + 1) // 2, ((w + 1) //2 + 1) // 2
            self.layers.append(nn.Flatten())
            self.layers.append(nn.Linear(x * y * 64, 512))
            self.layers.append(nn.GELU())
            self.layers.append(nn.Linear(512, n_actions))
        elif len(input_size) == 1:
            self.layers.append(nn.Linear(input_size[0], 128))
            self.layers.append(nn.GELU())
            self.layers.append(nn.Linear(128, 128))
            self.layers.append(nn.GELU())
            self.layers.append(nn.Linear(128, n_actions))
        
        self.layers.append(nn.Softmax(dim=-1))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.unsqueeze:
            x = x.unsqueeze(1)
        return self.layers(x)
    

def compute_cumulative_returns(rewards, gamma):
    n = len(rewards)

    # sum of discounted rewards
    powers = gamma ** torch.arange(n, dtype=torch.float32).to(rewards.device)
    weight_matrix = torch.triu(torch.outer(1 / powers, powers)).to(rewards.device)
    returns = torch.matmul(weight_matrix, rewards)

    # discounted rewards
    # discounts = torch.tensor([gamma ** i for i in range(n)]).to(rewards.device)
    # returns = rewards * discounts
    
    return returns


def reinforce_loss(logits, actions, returns, entropy_coef=0.01):
    log_probs = torch.log(logits)
    entropy = -(logits * log_probs).sum(dim=1).squeeze(-1)
    actions = actions.unsqueeze(1)
    selected_log_probs = log_probs.gather(1, actions).squeeze(1)
    pg_loss = -(selected_log_probs * returns).mean()
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

    model.train(True)
    model.to(device)
    
    with tqdm(range(start_step, steps + start_step)) as pbar:
        for step in pbar:
            if done:
                state, _ = env.reset()
            
            state_t = torch.tensor(state).float().to(device)
            with torch.no_grad():
                policy = model(state_t)
            
            action = strategy.select_action(policy.detach().cpu().numpy(), sample=True, softmax=True)
            # action = torch.multinomial(policy, 1).item()
            next_state, reward, done, out_of_bounds, _ = env.step(action)
            if done or out_of_bounds:
                done = True
                # reward = -10
            if tb_writer is not None:
                tb_writer.add_scalar('Global/reward', reward, step+1)

            # episode_reward = reward + discount_factor * episode_reward
            episode_reward += reward
            episode_steps += 1
            count += 1
            replay_memory.push(state, action, episode_reward, next_state, done)

            batch = replay_memory.sample_episodes(batch_size)
            # batch = replay_memory.sample_episode()
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                
                states = list(map(lambda x: torch.tensor(x).float().to(device), states))
                actions = list(map(lambda x: torch.tensor(x).long().to(device), actions))
                logits = list(map(lambda x: model(x), states))
                
                rewards = list(map(lambda x: torch.tensor(x).float().to(device), rewards))
                returns = list(map(lambda x: compute_cumulative_returns(x, gamma), rewards))
                
                losses = list(map(lambda x, y, z: reinforce_loss(x, y, z), logits, actions, returns))
                loss = sum(losses) / len(losses)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 100)
                optimizer.step()

                batch_loss = loss.item()
                total_loss += (batch_loss - total_loss) / count
                episode_loss += (batch_loss - episode_loss) / episode_steps

                if scheduler is not None:
                    if tb_writer is not None:
                        tb_writer.add_scalar("Global/lr", scheduler.get_last_lr()[-1], step+1)
                    scheduler.step(step)

                if tb_writer is not None:
                    tb_writer.add_scalar("Global/loss", batch_loss, step+1)
                    
            
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
            
            pbar.set_description(f"step: {(step+1):3d}/{steps+start_step}, ep_reward: {er:.0f}, ep_loss: {el:.2f}, avg_loss: {(total_loss / count):.2e}")

            if (step+1) % save_every == 0:
                if save_path is not None:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'episode': episode_count,
                        'step': step+1,
                        'loss': total_loss,
                    }, save_path)
