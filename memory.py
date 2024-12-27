from collections import deque
from random import sample, randint
import itertools

class ReplayMemory:
    def __init__(self, capacity):
        self.position = 0
        self.memory = []
        self.episodes = deque()
        self.capacity = capacity
        self.avg_episode_length = 0


    def push(self, *args):
        if len(self.episodes) == 0:
            self.episodes.appendleft([])
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        sars = (*args,)
        done = sars[4]
        self.memory[self.position] = sars
        self.position = (self.position + 1) % self.capacity
        self.episodes[0].append(sars) # because we are appending to the left
        if len(self.episodes) > 0 and len(self.memory) > self.capacity and done:
            e = self.episodes.pop()
            self.avg_episode_length += (self.avg_episode_length - len(e)) / (len(self.episodes) - 1) if len(self.episodes) > 1 else 0
            last_episode_length = len(self.episodes[0])
            self.avg_episode_length += (last_episode_length - self.avg_episode_length) / len(self.episodes)
        elif len(self.episodes) > 0 and done:
            last_episode_length = len(self.episodes[0])
            self.avg_episode_length += (last_episode_length - self.avg_episode_length) / len(self.episodes)
            self.episodes.appendleft([])

    
    def get_avg_episode_length(self):
        return self.avg_episode_length


    def sample(self, batch_size, n_steps=1):
        if n_steps < 1 or len(self.memory) < batch_size:
            return None
        if n_steps == 1:
            samples = sample(self.memory, batch_size)
            return zip(*samples)
        else:
            eligible_episodes = {i: e for i, e in enumerate(self.episodes) if len(e) >= n_steps}
            if len(eligible_episodes) == 0:
                return None
            episodes_lengths = [len(e) for e in eligible_episodes.values()]
            episodes_indecies = sample(eligible_episodes.keys(), batch_size, counts=episodes_lengths)
            sample_indecies = [randint(0, len(e) - n_steps) for e in [eligible_episodes[i] for i in episodes_indecies]]
            samples = [zip(*eligible_episodes[i][j:j+n_steps]) for i, j in zip(episodes_indecies, sample_indecies)]
            return zip(*samples)
        

    def sample_episode(self):
        if len(self.episodes) < 2:
            return None
        complete_episodes = list(itertools.islice(self.episodes, 1, None))
        episode = sample(complete_episodes, 1)[0]
        return zip(*episode)
    
    
    def sample_episodes(self, batch_size):
        if len(self.episodes) < batch_size + 1:
            return None
        complete_episodes = list(itertools.islice(self.episodes, 1, None))
        episodes = sample(complete_episodes, batch_size)
        episodes = [zip(*e) for e in episodes]
        return zip(*episodes)
