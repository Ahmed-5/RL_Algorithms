import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import copy


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
            x, y = (h + 1) // 2, (w + 1) // 2
            self.layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())
            x, y = (x + 1) // 2, (y + 1) // 2
            self.layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())
            x, y = (x + 1) // 2, (y + 1) // 2
            self.layers.append(nn.Flatten())
            self.layers.append(nn.Linear(x * y * 128, 512))
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
        unsqueeze=False,
        gamma=0.99,
        tau=0.01,
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
    target_model.train(False)
    model.to(device)
    target_model.to(device)
    
    with tqdm(range(start_step, steps + start_step)) as pbar:
        for step in pbar:
            if done:
                state, _ = env.reset()
            
            state_t = torch.tensor(state).float().to(device)
            with torch.no_grad():
                if unsqueeze:
                    state_t = state_t.unsqueeze(0)
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
                n_steps = np.random.binomial(n_steps, 0.2) + 1

            batch = replay_memory.sample(batch_size, n_steps)
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)
                
                if n_steps > 1:
                    states = np.array([i[0] for i in states])
                    actions = np.array([i[0] for i in actions])
                    next_states = np.array([i[-1] for i in next_states])
                    dones = np.array([i[-1] for i in dones])
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
                    tb_writer.add_scalar("Global/n_steps", n_steps, step+1)

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
            
            pbar.set_description(f"ep_reward: {er:.0f}, ep_loss: {el:.2e}, avg_loss: {(total_loss / count):.2e}, ep_len: {es}, n_steps: {n_steps}")

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

def compute_td_returns(rewards, gamma, td_lambda):
    """
    Compute TD(λ) returns more efficiently.
    """
    n = len(rewards)
    if n == 0:
        return torch.tensor([], device=rewards.device)
    
    # Pre-calculate discount factors (avoids recomputation)
    discount_factors = (gamma*td_lambda) ** torch.arange(n, dtype=torch.float32, device=rewards.device)
    discount_factors *= (1-td_lambda)
    discount_factors[-1] += td_lambda ** n
    
    # More efficient matrix creation using triu indices
    indices = torch.triu_indices(n, n, device=rewards.device)
    values = torch.ones(indices.shape[1], device=rewards.device)
    for i in range(indices.shape[1]):
        row, col = indices[0, i], indices[1, i]
        values[i] = discount_factors[col] / discount_factors[row]
    
    weight_matrix = torch.sparse_coo_tensor(indices, values, (n, n)).to_dense()
    returns = torch.matmul(weight_matrix, rewards)
    
    return returns


def compute_td_returns(rewards, gamma, td_lambda):
    """
    Compute TD(λ) returns more efficiently.
    """
    n = len(rewards)
    if n == 0:
        return torch.tensor([], device=rewards.device)
    
    # Pre-calculate discount factors (avoids recomputation)
    discount_factors = (gamma*td_lambda) ** torch.arange(n, dtype=torch.float32, device=rewards.device)
    discount_factors *= (1-td_lambda)
    discount_factors[-1] += td_lambda ** n
    
    # More efficient matrix creation using triu indices
    indices = torch.triu_indices(n, n, device=rewards.device)
    values = torch.ones(indices.shape[1], device=rewards.device)
    for i in range(indices.shape[1]):
        row, col = indices[0, i], indices[1, i]
        values[i] = discount_factors[col] / discount_factors[row]
    
    weight_matrix = torch.sparse_coo_tensor(indices, values, (n, n)).to_dense()
    returns = torch.matmul(weight_matrix, rewards)
    
    return returns


def train_dqn_tdLambda(
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
        td_lambda=0.95,
        start_step=0,
        save_path=None,
        save_every=10,
        logger=None,  # Changed tb_writer to logger
        scheduler=None,
        target_model=None,
        target_update_freq=1000,
        grad_clip_value=100
    ):
    """
    Optimized DQN training with TD(λ) and abstract logging.
    
    Args:
        model: The neural network model
        steps: Number of training steps
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to run on (cpu or cuda)
        env: Environment to interact with
        replay_memory: Replay buffer for experience replay
        strategy: Action selection strategy
        batch_size: Batch size for training
        gamma: Discount factor
        td_lambda: TD(λ) parameter
        start_step: Step to start from (for resuming training)
        save_path: Path to save model checkpoints
        save_every: How often to save checkpoints
        logger: AbstractLogger instance for logging metrics
        scheduler: Learning rate scheduler
        target_model: Target network for stable learning
        target_update_freq: How often to update target network
        grad_clip_value: Value for gradient clipping
    """
    done = True
    total_reward = 0
    total_loss = 0
    episode_count = 0
    reward = 0
    episode_reward = 0
    episode_steps = 0
    episode_loss = 0
    count = 0
    
    # Create target model if not provided
    if target_model is None:
        target_model = copy.deepcopy(model)
    target_model.to(device)
    target_model.eval()
    
    model.train(True)
    model.to(device)
    
    # Log hyperparameters if logger is provided
    if logger is not None:
        hyperparams = {
            'gamma': gamma,
            'td_lambda': td_lambda,
            'batch_size': batch_size,
            'grad_clip_value': grad_clip_value,
            'target_update_freq': target_update_freq,
            'optimizer': optimizer.__class__.__name__,
            'criterion': criterion.__class__.__name__,
        }
        logger.log_hyperparams(hyperparams)
    
    with tqdm(range(start_step, steps + start_step)) as pbar:
        for step in pbar:
            if done:
                state, _ = env.reset()
            
            # Convert state to tensor
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            
            # Get action from policy
            with torch.no_grad():
                policy = model(state_t)
            
            action = strategy.select_action(policy.detach().cpu().numpy(), sample=True, softmax=True)
            next_state, reward, done, out_of_bounds, _ = env.step(action)
            
            if done or out_of_bounds:
                done = True
            
            # Log reward
            if logger is not None:
                logger.log_scalar('Global/reward', reward, step+1)
            
            # Update episode statistics
            episode_reward += reward
            episode_steps += 1
            count += 1
            
            # Store experience
            replay_memory.push(state, action, reward, next_state, done)
            
            # Sample from replay buffer
            batch = replay_memory.sample_episodes(batch_size)
            
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                
                # Batch processing - convert to tensors
                states_batch = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                actions_batch = torch.tensor(np.array(actions), dtype=torch.long, device=device)
                rewards_batch = [torch.tensor(r, dtype=torch.float32, device=device) for r in rewards]
                next_states_batch = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones_masks = [torch.tensor([0 if i else gamma**(ind+1) for ind, i in enumerate(d)], device=device) for d in dones]
                
                # Compute returns for each episode in batch
                returns = [compute_td_returns(r, gamma, td_lambda) for r in rewards_batch]
                returns_batch = torch.cat(returns)
                
                # Get next state values using target network
                with torch.no_grad():
                    next_q_values = target_model(next_states_batch).max(1)[0]
                
                # Calculate target values
                masks_batch = torch.cat(dones_masks)
                target_values = returns_batch + masks_batch * next_q_values
                
                # Get current Q-values
                current_q_values = model(states_batch).gather(1, actions_batch.unsqueeze(1))
                
                # Calculate loss
                loss = criterion(current_q_values, target_values.unsqueeze(1))
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
                optimizer.step()
                
                # Update statistics
                batch_loss = loss.item()
                total_loss += batch_loss
                episode_loss += batch_loss
                
                # Update learning rate
                if scheduler is not None:
                    if logger is not None:
                        logger.log_scalar("Global/lr", scheduler.get_last_lr()[-1], step+1)
                    scheduler.step()
                
                # Log metrics
                if logger is not None:
                    logger.log_scalar("Global/loss", batch_loss, step+1)
                    
                    # Log parameter histograms periodically
                    if step % 100 == 0:
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                logger.log_histogram(f"Parameters/{name}", param.data.cpu().numpy(), step+1)
                                if param.grad is not None:
                                    logger.log_histogram(f"Gradients/{name}", param.grad.data.cpu().numpy(), step+1)
                
                # Update target network
                if step % target_update_freq == 0:
                    target_model.load_state_dict(model.state_dict())
            
            # Episode completion
            if done:
                if logger is not None:
                    avg_episode_loss = episode_loss / max(1, episode_steps)
                    logger.log_scalar('Episode/reward', episode_reward, episode_count+1)
                    logger.log_scalar('Episode/steps', episode_steps, episode_count+1)
                    logger.log_scalar('Episode/loss', avg_episode_loss, episode_count+1)
                    logger.flush()  # Ensure logs are written
                
                # Update episode stats
                episode_count += 1
                total_reward += episode_reward
                
                # Store for progress bar
                er = episode_reward
                es = episode_steps
                el = episode_loss / max(1, episode_steps)
                
                # Reset episode variables
                episode_reward = 0
                episode_steps = 0
                episode_loss = 0
            else:
                state = next_state
            
            # Update progress bar
            avg_loss = total_loss / max(1, count)
            avg_reward = total_reward / max(1, episode_count)
            pbar.set_description(
                f"step: {(step+1):3d}/{steps+start_step}, "
                f"ep_reward: {er:.1f}, ep_loss: {el:.2f}, "
                f"avg_loss: {avg_loss:.2e}, avg_reward: {avg_reward:.1f}"
            )
            
            # Save model checkpoint
            if save_path is not None and (step+1) % save_every == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'target_model_state_dict': target_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode_count,
                    'step': step+1,
                    'loss': total_loss / max(1, count),
                    'reward': total_reward / max(1, episode_count),
                }
                torch.save(checkpoint, save_path)
                
                if logger is not None:
                    logger.log_text("Checkpoints/info", f"Saved checkpoint at step {step+1}", step+1)
    
    # Final cleanup
    if logger is not None:
        logger.flush()
    
    return model, target_model, {
        'episodes': episode_count,
        'steps': steps,
        'avg_reward': total_reward / max(1, episode_count),
        'avg_loss': total_loss / max(1, count)
    }