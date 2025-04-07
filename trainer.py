from abc import abstractmethod
import torch
from tqdm import tqdm


class Trainer:
    """Base trainer class for RL Algorithms with common functionality"""
    
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

