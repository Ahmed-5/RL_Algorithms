import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN, train_dqn, train_dqn_target, train_ddqn
from REINFORCE import REINFORCE, train_reinforce
# from a2c import Actor, Critic, train_a2c
from ppo import Actor, Critic, train_ppo
from memory import ReplayMemory
from strategy import EpsilonGreedyStrategy, ExponentialGreedyStrategy
import gymnasium as gym
import ale_py
from tensorboardX import SummaryWriter

# gym.register_envs(ale_py)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
steps = 20000
n_steps = 1
lr = 1e-4
batch_size = 256
reinforce_batch = 1

sw = SummaryWriter()
# ENV_NAME = 'CartPole-v1'
ENV_NAME = 'ALE/AirRaid-v5'
env = gym.make(ENV_NAME)
input_shape = env.observation_space.shape
unsqueeze = len(input_shape) != 1
num_actions = env.action_space.n

adaptive_n_steps=False

model = DQN(input_shape, num_actions)
target_model = DQN(input_shape, num_actions)

# model = REINFORCE(input_shape, num_actions)

# actor = Actor(input_shape, num_actions)
# critic = Critic(input_shape)

memory = ReplayMemory(10000)
# strategy = EpsilonGreedyStrategy(epsilon=1.0, decay=0.99, min_epsilon=0.01)
strategy = ExponentialGreedyStrategy(epsilon=1.0, decay=0.99, min_epsilon=0.01)

optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
# optimizer = optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=lr, amsgrad=True)
loss_function = nn.MSELoss()

# train_dqn(
#     model, steps, optimizer, loss_function, 
#     device, env, memory, strategy, 
#     batch_size=batch_size, gamma=0.9, n_steps=n_steps, adaptive_n_steps=adaptive_n_steps, tb_writer=sw
# )

train_dqn_target(
    model, target_model, steps, optimizer, loss_function, 
    device, env, memory, strategy, batch_size=batch_size, unsqueeze=unsqueeze,
    gamma=0.9, n_steps=n_steps, adaptive_n_steps=adaptive_n_steps, tb_writer=sw
)

# train_ddqn(
#     model, target_model, steps, optimizer, loss_function, 
#     device, env, memory, strategy, 
#     batch_size=batch_size, gamma=0.9, n_steps=n_steps, tb_writer=sw
# )

# train_reinforce(
#     model, steps, optimizer, 
#     device, env, memory, strategy, 
#     batch_size=reinforce_batch, gamma=0.9, tb_writer=sw
# )

# train_a2c(
#     actor, critic, steps, optimizer, loss_function, 
#     device, env, memory, strategy, 
#     batch_size=batch_size, gamma=0.9, n_steps=n_steps, tb_writer=sw
# )

# train_ppo(
#     actor, critic, steps, optimizer, loss_function, 
#     device, env, memory, strategy, 
#     batch_size=batch_size, gamma=0.9, ppo_clip=0.2, n_steps=n_steps, tb_writer=sw
# )

env.close()