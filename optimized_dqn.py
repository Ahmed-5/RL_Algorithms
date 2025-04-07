import numpy as np
import torch
from torch import nn
import copy
from utilities import compute_td_returns
from trainer import Trainer

class VanillaDQNTrainer(Trainer):
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


class DQNWithTargetNetTrainer(Trainer):
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