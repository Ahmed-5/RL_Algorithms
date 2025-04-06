import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import copy
from abc import ABC, abstractmethod


class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.unsqueeze = False

        self.layers = []
        if len(input_size) == 3 or len(input_size) == 2:
            self._build_conv_network(input_size)
        elif len(input_size) == 1:
            self._build_fc_network(input_size)

        self.layers = nn.Sequential(*self.layers)

    def _build_conv_network(self, input_size):
        self.unsqueeze = len(input_size) == 2
        channels = 1 if len(input_size) == 2 else input_size[0]
        h = input_size[-2]
        w = input_size[-1]
        
        # First conv block
        self.layers.append(nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())
        x, y = (h + 1) // 2, (w + 1) // 2
        
        # Second conv block
        self.layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())
        x, y = (x + 1) // 2, (y + 1) // 2
        
        # Third conv block
        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())
        x, y = (x + 1) // 2, (y + 1) // 2
        
        # FC layers
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(x * y * 128, 512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(512, self.n_actions))

    def _build_fc_network(self, input_size):
        self.layers.append(nn.Linear(input_size[0], 128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(128, 128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(128, self.n_actions))

    def forward(self, x):
        if self.unsqueeze:
            x = x.unsqueeze(1)
        return self.layers(x)


class AbstractLogger(ABC):
    """Abstract base class for logging during training"""
    
    @abstractmethod
    def log_scalar(self, tag, value, step):
        """Log a scalar value"""
        pass
        
    @abstractmethod
    def log_histogram(self, tag, values, step):
        """Log a histogram of values"""
        pass
        
    @abstractmethod
    def log_text(self, tag, text, step):
        """Log text"""
        pass
        
    @abstractmethod
    def log_hyperparams(self, params_dict):
        """Log hyperparameters"""
        pass
        
    @abstractmethod
    def flush(self):
        """Ensure all logs are written"""
        pass


class TensorboardLogger(AbstractLogger):
    """TensorBoard implementation of logger"""
    
    def __init__(self, writer):
        self.writer = writer
        
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        
    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)
        
    def log_text(self, tag, text, step):
        self.writer.add_text(tag, text, step)
        
    def log_hyperparams(self, params_dict):
        # TensorBoard's add_hparams requires a bit more setup
        # For simplicity, we'll just log each param as a scalar
        for param_name, param_value in params_dict.items():
            if isinstance(param_value, (int, float)):
                self.writer.add_scalar(f"Hyperparams/{param_name}", param_value, 0)
            else:
                self.writer.add_text(f"Hyperparams/{param_name}", str(param_value), 0)
        
    def flush(self):
        self.writer.flush()


class ReplayBuffer:
    """Experience replay buffer for RL agents"""
    
    def __init__(self, capacity, max_episode_length=1000):
        self.capacity = capacity
        self.max_episode_length = max_episode_length
        self.buffer = []
        self.position = 0
        self.episode_buffer = []
        self.episode_lengths = []
        
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.episode_buffer.append((state, action, reward, next_state, done))
        
        if done:
            # Handle episode completion
            self.episode_lengths.append(len(self.episode_buffer))
            if len(self.episode_lengths) > 100:  # Keep statistics for last 100 episodes
                self.episode_lengths.pop(0)
                
            # Add episode to main buffer
            if len(self.buffer) < self.capacity:
                self.buffer.append(self.episode_buffer)
            else:
                self.buffer[self.position] = self.episode_buffer
                self.position = (self.position + 1) % self.capacity
                
            self.episode_buffer = []
            
    def get_avg_episode_length(self):
        """Get average episode length"""
        if not self.episode_lengths:
            return self.max_episode_length  # Default if no episodes completed yet
        return sum(self.episode_lengths) / len(self.episode_lengths)
        
    def sample(self, batch_size, n_steps=1):
        """Sample batch of transitions"""
        if len(self.buffer) < batch_size // 10:  # Need some minimum buffer size
            return None
            
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        
        # Sample episodes
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        
        for idx in indices:
            episode = self.buffer[idx]
            if len(episode) < n_steps:
                continue
                
            start_idx = np.random.randint(0, len(episode) - n_steps + 1)
            sequence = episode[start_idx:start_idx + n_steps]
            
            states, actions, rewards, next_states, dones = zip(*sequence)
            
            state_batch.append(states)
            action_batch.append(actions)
            reward_batch.append(rewards)
            next_state_batch.append(next_states)
            done_batch.append(dones)
            
        if not state_batch:  # If we couldn't sample enough valid sequences
            return None
            
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        
    def sample_episodes(self, batch_size):
        """Sample complete episodes for algorithms that work on episode data"""
        if len(self.buffer) < batch_size // 10:
            return None
            
        episodes = np.random.choice(self.buffer, min(batch_size, len(self.buffer)), replace=False)
        
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for episode in episodes:
            states, actions, rewards, next_states, dones = zip(*episode)
            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
            
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch


class ActionStrategy:
    """Base class for action selection strategies"""
    
    @abstractmethod
    def select_action(self, q_values, **kwargs):
        """Select action based on q-values"""
        pass


class EpsilonGreedyStrategy(ActionStrategy):
    """Epsilon-greedy action selection strategy"""
    
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        
    def select_action(self, q_values, sample=False, softmax=False):
        """Select action using epsilon-greedy policy"""
        self.steps += 1
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-self.steps / self.epsilon_decay)
        
        if sample and np.random.random() < epsilon:
            # Random action
            if softmax:
                # Use softmax distribution
                probs = F.softmax(torch.tensor(q_values), dim=-1).numpy()
                return np.random.choice(len(q_values), p=probs)
            else:
                # Uniform random
                return np.random.randint(0, len(q_values))
        else:
            # Greedy action
            return np.argmax(q_values)


def compute_td_returns(rewards, gamma, td_lambda):
    """
    Compute TD(λ) returns efficiently.
    
    Args:
        rewards: Tensor of rewards for an episode
        gamma: Discount factor
        td_lambda: TD(λ) parameter
        
    Returns:
        Tensor of TD(λ) returns
    """
    n = len(rewards)
    if n == 0:
        return torch.tensor([], device=rewards.device)
    
    # Pre-calculate discount factors
    discount_factors = (gamma * td_lambda) ** torch.arange(n, dtype=torch.float32, device=rewards.device)
    discount_factors *= (1 - td_lambda)
    discount_factors[-1] += td_lambda ** n
    
    # Create weight matrix using triu indices
    indices = torch.triu_indices(n, n, device=rewards.device)
    values = torch.ones(indices.shape[1], device=rewards.device)
    
    for i in range(indices.shape[1]):
        row, col = indices[0, i], indices[1, i]
        values[i] = discount_factors[col] / discount_factors[row]
    
    weight_matrix = torch.sparse_coo_tensor(indices, values, (n, n)).to_dense()
    returns = torch.matmul(weight_matrix, rewards)
    
    return returns


class DQNTrainer:
    """Base DQN trainer class with common functionality"""
    
    def __init__(
        self,
        model,
        env,
        replay_buffer,
        action_strategy,
        optimizer,
        criterion,
        device,
        gamma=0.99,
        grad_clip_value=100,
        save_path=None,
        save_every=10,
        logger=None
    ):
        self.model = model
        self.env = env
        self.replay_buffer = replay_buffer
        self.action_strategy = action_strategy
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.gamma = gamma
        self.grad_clip_value = grad_clip_value
        self.save_path = save_path
        self.save_every = save_every
        self.logger = logger
        
        # Training state
        self.episode_count = 0
        self.total_reward = 0
        self.total_loss = 0
        self.count = 0
        
        # Current episode state
        self.state = None
        self.done = True
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_loss = 0
        
    def _reset_if_done(self):
        """Reset environment if episode is done"""
        if self.done:
            self.state, _ = self.env.reset()
            return True
        return False
        
    def _select_action(self):
        """Select action based on current state"""
        state_t = torch.tensor(self.state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return self.action_strategy.select_action(q_values.detach().cpu().numpy())
        
    def _step_environment(self, action):
        """Take step in environment"""
        next_state, reward, done, out_of_bounds, _ = self.env.step(action)
        
        if done or out_of_bounds:
            done = True
            
        # Update episode tracking
        self.episode_reward += reward
        self.episode_steps += 1
        self.count += 1
        
        # Log reward if logger available
        if self.logger is not None:
            self.logger.log_scalar('Global/reward', reward, self.count)
            
        # Store transition
        self.replay_buffer.push(self.state, action, reward, next_state, done)
        
        return next_state, reward, done
        
    def _handle_episode_completion(self):
        """Handle logic when episode completes"""
        if self.done:
            # Update stats
            self.episode_count += 1
            self.total_reward += self.episode_reward
            
            # Log episode metrics
            if self.logger is not None:
                avg_episode_loss = self.episode_loss / max(1, self.episode_steps)
                self.logger.log_scalar('Episode/reward', self.episode_reward, self.episode_count)
                self.logger.log_scalar('Episode/steps', self.episode_steps, self.episode_count)
                self.logger.log_scalar('Episode/loss', avg_episode_loss, self.episode_count)
                
            # Store values for progress bar
            er = self.episode_reward
            es = self.episode_steps
            el = self.episode_loss / max(1, self.episode_steps)
            
            # Reset episode variables
            self.episode_reward = 0
            self.episode_steps = 0
            self.episode_loss = 0
            
            return er, es, el
        return None, None, None
        
    def _save_checkpoint(self, step):
        """Save model checkpoint if needed"""
        if self.save_path is not None and (step+1) % self.save_every == 0:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'episode': self.episode_count,
                'step': step+1,
                'loss': self.total_loss / max(1, self.count),
                'reward': self.total_reward / max(1, self.episode_count),
            }
            torch.save(checkpoint, self.save_path)
            
            if self.logger is not None:
                self.logger.log_text("Checkpoints/info", f"Saved checkpoint at step {step+1}", step+1)
                
    @abstractmethod
    def train_step(self, batch):
        """Implement training step for specific algorithm"""
        pass
        
    def _update_progress_bar(self, pbar, step, steps, start_step, er, es, el):
        """Update progress bar with current stats"""
        avg_loss = self.total_loss / max(1, self.count)
        avg_reward = self.total_reward / max(1, self.episode_count)
        
        pbar.set_description(
            f"step: {(step+1):3d}/{steps+start_step}, "
            f"ep_reward: {er:.1f}, ep_loss: {el:.2f}, "
            f"avg_loss: {avg_loss:.2e}, avg_reward: {avg_reward:.1f}"
        )
        
    def train(self, steps, batch_size, start_step=0, scheduler=None):
        """Main training loop"""
        # Set up training
        self.model.train(True)
        self.model.to(self.device)
        
        # Initialize or reset tracking variables
        self.done = True
        er, es, el = 0, 0, 0
        
        with tqdm(range(start_step, steps + start_step)) as pbar:
            for step in pbar:
                # Reset if needed
                self._reset_if_done()
                
                # Select action and step environment
                action = self._select_action()
                next_state, reward, done = self._step_environment(action)
                self.done = done
                
                # Get batch and train
                batch = self.replay_buffer.sample(batch_size)
                if batch is not None:
                    loss = self.train_step(batch)
                    
                    # Update statistics
                    self.total_loss += loss
                    self.episode_loss += loss
                    
                    # Update learning rate
                    if scheduler is not None:
                        if self.logger is not None:
                            self.logger.log_scalar("Global/lr", scheduler.get_last_lr()[-1], step+1)
                        scheduler.step()
                
                # Update state
                if not self.done:
                    self.state = next_state
                
                # Handle episode completion
                episode_complete = self._handle_episode_completion()
                if episode_complete[0] is not None:
                    er, es, el = episode_complete
                
                # Update progress bar
                self._update_progress_bar(pbar, step, steps, start_step, er, es, el)
                
                # Save checkpoint
                self._save_checkpoint(step)
        
        # Final cleanup
        if self.logger is not None:
            self.logger.flush()
            
        return {
            'episodes': self.episode_count,
            'steps': steps,
            'avg_reward': self.total_reward / max(1, self.episode_count),
            'avg_loss': self.total_loss / max(1, self.count)
        }


class VanillaDQNTrainer(DQNTrainer):
    """Vanilla DQN implementation"""
    
    def train_step(self, batch):
        """Training step for vanilla DQN"""
        states, actions, rewards, next_states, dones = batch
        
        # Process batch data
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Handle n-step returns if needed
        if isinstance(states[0], list) or isinstance(states[0], tuple):
            states = [i[0] for i in states]
            actions = [i[0] for i in actions]
            next_states = [i[-1] for i in next_states]
            dones = [i[-1] for i in dones]
            
            # Calculate n-step return
            n_steps = len(rewards[0])
            gammas = np.array([self.gamma ** i for i in range(n_steps)])
            rewards = np.sum(rewards * gammas, axis=1)
        
        # Convert to tensors
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        next_state_masks = torch.tensor([0 if i else 1 for i in dones]).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1).values
        
        # Calculate target values
        n_steps = 1 if not isinstance(states[0], list) else len(rewards[0])
        labels = rewards + (self.gamma ** n_steps) * next_state_masks * next_q_values
        labels = labels.unsqueeze(1).to(self.device)
        
        # Get current Q-values
        logits = self.model(states).gather(1, actions)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)
        self.optimizer.step()
        
        return loss.item()


class DQNWithTargetNetTrainer(DQNTrainer):
    """DQN with target network"""
    
    def __init__(self, *args, tau=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(self.device)
        self.target_model.eval()
        
    def _update_target_network(self):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def train_step(self, batch):
        """Training step with target network"""
        states, actions, rewards, next_states, dones = batch
        
        # Process batch data
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Handle n-step returns if needed
        if isinstance(states[0], list) or isinstance(states[0], tuple):
            states = np.array([i[0] for i in states])
            actions = np.array([i[0] for i in actions])
            next_states = np.array([i[-1] for i in next_states])
            dones = np.array([i[-1] for i in dones])
            
            # Calculate n-step return
            n_steps = len(rewards[0])
            gammas = np.array([self.gamma ** i for i in range(n_steps)])
            rewards = np.sum(rewards * gammas, axis=1)
        
        # Convert to tensors
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        next_state_masks = torch.tensor([0 if i else 1 for i in dones]).to(self.device)
        
        # Get Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1).values
        
        # Calculate target values
        n_steps = 1 if not isinstance(states[0], list) else len(rewards[0])
        labels = rewards + (self.gamma ** n_steps) * next_state_masks * next_q_values
        labels = labels.unsqueeze(1).to(self.device)
        
        # Get current Q-values
        logits = self.model(states).gather(1, actions)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)
        self.optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        return loss.item()


class DoubleDQNTrainer(DQNWithTargetNetTrainer):
    """Double DQN implementation"""
    
    def train_step(self, batch):
        """Training step for Double DQN"""
        states, actions, rewards, next_states, dones = batch
        
        # Process batch data
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Handle n-step returns if needed
        if isinstance(states[0], list) or isinstance(states[0], tuple):
            states = [i[0] for i in states]
            actions = [i[0] for i in actions]
            next_states = [i[-1] for i in next_states]
            dones = [i[-1] for i in dones]
            
            # Calculate n-step return
            n_steps = len(rewards[0])
            gammas = np.array([self.gamma ** i for i in range(n_steps)])
            rewards = np.sum(rewards * gammas, axis=1)
        
        # Convert to tensors
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        next_state_masks = torch.tensor([0 if i else 1 for i in dones]).to(self.device)
        
        # Double DQN: select actions using online network
        with torch.no_grad():
            next_actions = self.model(next_states).max(1).indices
            # Evaluate actions using target network
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target values
        n_steps = 1 if not isinstance(states[0], list) else len(rewards[0])
        labels = rewards + (self.gamma ** n_steps) * next_state_masks * next_q_values
        labels = labels.unsqueeze(1).to(self.device)
        
        # Get current Q-values
        logits = self.model(states).gather(1, actions)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)
        self.optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        return loss.item()


class TDLambdaDQNTrainer(DQNWithTargetNetTrainer):
    """DQN with TD(λ) returns"""
    
    def __init__(self, *args, td_lambda=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.td_lambda = td_lambda
    
    def train_step(self, batch):
        """Training step using TD(λ) returns"""
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = batch
        
        # Convert episode data to tensors
        states_batch = torch.tensor(np.array(states_batch), dtype=torch.float32, device=self.device)
        actions_batch = torch.tensor(np.array(actions_batch), dtype=torch.long, device=self.device)
        rewards_batch = [torch.tensor(r, dtype=torch.float32, device=self.device) for r in rewards_batch]
        next_states_batch = torch.tensor(np.array(next_states_batch), dtype=torch.float32, device=self.device)
        dones_masks = [torch.tensor([0 if i else self.gamma**(ind+1) for ind, i in enumerate(d)], 
                                     device=self.device) for d in dones_batch]
        
        # Compute TD(λ) returns for each episode
        returns = [compute_td_returns(r, self.gamma, self.td_lambda) for r in rewards_batch]
        returns_batch = torch.cat(returns)
        
        # Get next state values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states_batch).max(1)[0]
        
        # Calculate target values
        masks_batch = torch.cat(dones_masks)
        target_values = returns_batch + masks_batch * next_q_values
        
        # Get current Q-values
        current_q_values = self.model(states_batch).gather(1, actions_batch.unsqueeze(1))
        
        # Calculate loss
        loss = self.criterion(current_q_values, target_values.unsqueeze(1))
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)
        self.optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        return loss.item()


# Adapter functions to maintain compatibility with original code
def train_dqn(model, steps, optimizer, criterion, device, env, replay_memory, strategy, batch_size, **kwargs):
    """Adapter function for vanilla DQN"""
    trainer = VanillaDQNTrainer(
        model=model, 
        env=env, 
        replay_buffer=replay_memory, 
        action_strategy=strategy,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        **kwargs
    )
    return trainer.train(steps, batch_size, **kwargs)


def train_dqn_target(model, target_model, steps, optimizer, criterion, device, env, replay_memory, strategy, batch_size, **kwargs):
    """Adapter function for DQN with target network"""
    trainer = DQNWithTargetNetTrainer(
        model=model,
        env=env,
        replay_buffer=replay_memory,
        action_strategy=strategy,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        **kwargs
    )
    # Use provided target model
    trainer.target_model = target_model
    return trainer.train(steps, batch_size, **kwargs)


def train_ddqn(model, target_model, steps, optimizer, criterion, device, env, replay_memory, strategy, batch_size, **kwargs):
    """Adapter function for Double DQN"""
    trainer = DoubleDQNTrainer(
        model=model,
        env=env,
        replay_buffer=replay_memory,
        action_strategy=strategy,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        **kwargs
    )
    # Use provided target model
    trainer.target_model = target_model
    return trainer.train(steps, batch_size, **kwargs)


def train_dqn_tdLambda(model, steps, optimizer, criterion, device, env, replay_memory, strategy, batch_size, **kwargs):
    """Adapter function for DQN with TD(λ)"""
    trainer = TDLambdaDQNTrainer(
        model=model,
        env=env,
        replay_buffer=replay_memory,
        action_strategy=strategy,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        **kwargs
    )
    return trainer.train(steps, batch_size, **kwargs)