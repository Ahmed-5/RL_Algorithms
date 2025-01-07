import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()
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
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            x, y = ((h + 1 // 2) + 1) // 2, ((w + 1) //2 + 1) // 2
            self.layers.append(nn.Flatten())
            self.layers.append(nn.Linear(x * y * 64, 512))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(512, n_actions))
        elif len(input_size) == 1:
            self.layers.append(nn.Linear(input_size[0], 128))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(128, 128))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(128, n_actions))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.unsqueeze:
            x = x.unsqueeze(1)
        return self.layers(x)


def train_dqn(
        model, 
        steps, 
        optimizer, 
        criterion, 
        device, 
        env,
        replay_memory, 
        strategy,
        batch_size,
        gamma=0.99,
        n_steps=1,
        adaptive_n_steps=False,
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
                q_values = model(state_t)
            
            action = strategy.select_action(q_values.detach().cpu().numpy())
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
            replay_memory.push(state, action, reward, next_state, done)

            if adaptive_n_steps:
                n_steps = replay_memory.get_avg_episode_length()
                n_steps = max(1, np.sqrt(n_steps)) - 1
                n_steps = np.random.binomial(n_steps, 0.3) + 1

            batch = replay_memory.sample(batch_size, n_steps)
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)
                
                if n_steps > 1:
                    states = [i[0] for i in states]
                    actions = [i[0] for i in actions]
                    next_states = [i[-1] for i in next_states]
                    dones = [i[-1] for i in dones]
                    rewards = np.array(rewards)
                    gammas = np.array([gamma ** i for i in range(n_steps)])
                    rewards = np.sum(rewards * gammas, axis=1)

                states = torch.tensor(states).float().to(device)
                actions = torch.tensor(actions).long().unsqueeze(1).to(device)
                rewards = torch.tensor(rewards).float().to(device)
                next_states = torch.tensor(next_states).float().to(device)
                next_state_masks = torch.tensor([0 if i else 1 for i in dones]).to(device)
                
                with torch.no_grad():
                    next_q_values = model(next_states).max(1).values
                labels = rewards + (gamma ** n_steps) * next_state_masks * next_q_values
                labels = labels.unsqueeze(1).to(device)

                logits = model(states).gather(1, actions)
                loss = criterion(logits, labels)

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

                    
def train_dqn_target(
        model, 
        target_model,
        steps, 
        optimizer, 
        criterion, 
        device, 
        env,
        replay_memory, 
        strategy,
        batch_size,
        gamma=0.99,
        tau=0.01,
        n_steps=1,
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
    target_model.train(False)
    model.to(device)
    target_model.to(device)
    
    with tqdm(range(start_step, steps + start_step)) as pbar:
        for step in pbar:
            if done:
                state, _ = env.reset()
            
            state_t = torch.tensor(state).float().to(device)
            with torch.no_grad():
                q_values = model(state_t)
            
            action = strategy.select_action(q_values.detach().cpu().numpy())
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
            replay_memory.push(state, action, reward, next_state, done)

            batch = replay_memory.sample(batch_size, n_steps)
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)
                
                if n_steps > 1:
                    states = [i[0] for i in states]
                    actions = [i[0] for i in actions]
                    next_states = [i[-1] for i in next_states]
                    dones = [i[-1] for i in dones]
                    rewards = np.array(rewards)
                    gammas = np.array([gamma ** i for i in range(n_steps)])
                    rewards = np.sum(rewards * gammas, axis=1)

                states = torch.tensor(states).float().to(device)
                actions = torch.tensor(actions).long().unsqueeze(1).to(device)
                rewards = torch.tensor(rewards).float().to(device)
                next_states = torch.tensor(next_states).float().to(device)
                next_state_masks = torch.tensor([0 if i else 1 for i in dones]).to(device)
                
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1).values
                labels = rewards + (gamma ** n_steps) * next_state_masks * next_q_values
                labels = labels.unsqueeze(1).to(device)

                logits = model(states).gather(1, actions)
                loss = criterion(logits, labels)

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

                for target_param, param in zip(target_model.parameters(), model.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                    
            
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


def train_ddqn(
        model, 
        target_model,
        steps, 
        optimizer, 
        criterion, 
        device, 
        env,
        replay_memory, 
        strategy,
        batch_size,
        gamma=0.99,
        tau=0.01,
        n_steps=1,
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
    target_model.train(False)
    model.to(device)
    target_model.to(device)
    
    with tqdm(range(start_step, steps + start_step)) as pbar:
        for step in pbar:
            if done:
                state, _ = env.reset()
            
            state_t = torch.tensor(state).float().to(device)
            with torch.no_grad():
                q_values = model(state_t)
            
            action = strategy.select_action(q_values.detach().cpu().numpy())
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
            replay_memory.push(state, action, reward, next_state, done)

            batch = replay_memory.sample(batch_size, n_steps)
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)
                
                if n_steps > 1:
                    states = [i[0] for i in states]
                    actions = [i[0] for i in actions]
                    next_states = [i[-1] for i in next_states]
                    dones = [i[-1] for i in dones]
                    rewards = np.array(rewards)
                    gammas = np.array([gamma ** i for i in range(n_steps)])
                    rewards = np.sum(rewards * gammas, axis=1)

                states = torch.tensor(states).float().to(device)
                actions = torch.tensor(actions).long().unsqueeze(1).to(device)
                rewards = torch.tensor(rewards).float().to(device)
                next_states = torch.tensor(next_states).float().to(device)
                next_state_masks = torch.tensor([0 if i else 1 for i in dones]).to(device)
                
                with torch.no_grad():
                    next_actions = model(next_states).max(1).indices
                    next_q_values = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                labels = rewards + (gamma ** n_steps) * next_state_masks * next_q_values
                labels = labels.unsqueeze(1).to(device)

                logits = model(states).gather(1, actions)
                loss = criterion(logits, labels)

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

                for target_param, param in zip(target_model.parameters(), model.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                    
            
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
