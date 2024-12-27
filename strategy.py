import numpy as np
import random
import math

class Strategy:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_action(self, q_values, sample=False, softmax=False):
        if random.random() < self.epsilon:
            action = random.randint(0, len(q_values) - 1)
        else:
            if sample and not softmax:
                action = np.random.choice(len(q_values), p=np.exp(q_values) / np.sum(np.exp(q_values)))
            elif sample and softmax:
                action = np.random.choice(len(q_values), p=q_values)
            else:
                action = np.argmax(q_values)

        self.decay_epsilon()
        return action
    
    def decay_epsilon(self):
        raise NotImplementedError

# class EpsilonGreedyStrategy:
#     def __init__(self, epsilon: float, decay: float = 0.99, min_epsilon: float = 0.01):
#         self.epsilon = epsilon
#         self.decay = decay
#         self.min_epsilon = min_epsilon

#     def select_action(self, q_values):
#         if random.random() < self.epsilon:
#             action = random.randint(0, len(q_values) - 1)
#         else:
#             action = np.argmax(q_values)

#         self.decay_epsilon()
#         return action

#     def decay_epsilon(self):
#         self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

class EpsilonGreedyStrategy(Strategy):
    def __init__(self, epsilon: float, decay: float = 0.99, min_epsilon: float = 0.01):
        super().__init__(epsilon)
        self.decay = decay
        self.min_epsilon = min_epsilon

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


# class ExponentialGreedyStrategy:
#     def __init__(self, epsilon: float, decay: float = 0.99, min_epsilon: float = 0.01):
#         self.epsilon_start = epsilon
#         self.epsilon = epsilon
#         self.decay = decay
#         self.min_epsilon = min_epsilon
#         self.steps_done = 0

#     def select_action(self, q_values, sample=False, softmax=False):
#         if random.random() < self.epsilon:
#             action = random.randint(0, len(q_values) - 1)
#         else:
#             if sample and not softmax:
#                 action = np.random.choice(len(q_values), p=np.exp(q_values) / np.sum(np.exp(q_values)))
#             elif sample and softmax:
#                 action = np.random.choice(len(q_values), p=q_values)
#             else:
#                 action = np.argmax(q_values)

#         self.decay_epsilon()
#         return action

#     def decay_epsilon(self):
#         self.steps_done += 1
#         self.epsilon = self.min_epsilon + (self.epsilon_start - self.min_epsilon) * math.exp(-1. * self.steps_done / self.decay)


class ExponentialGreedyStrategy(Strategy):
    def __init__(self, epsilon: float, decay: float = 0.99, min_epsilon: float = 0.01):
        super().__init__(epsilon)
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.steps_done = 0

    def decay_epsilon(self):
        self.steps_done += 1
        self.epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * math.exp(-1. * self.steps_done / self.decay)
        
        
class SoftmaxStrategy:
    def __init__(self, temperature: float):
        self.temperature = temperature

    def select_action(self, q_values):
        exp_q = np.exp(q_values / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probabilities)


class UCBStrategy:
    def __init__(self, c: float):
        self.c = c
        self.counts = None
        self.values = None

    def initialize(self, n_actions):
        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)

    def select_action(self, q_values, total_steps):
        if self.counts is None:
            self.initialize(len(q_values))

        ucb_values = self.values + self.c * np.sqrt(np.log(total_steps + 1) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = value + (reward - value) / n