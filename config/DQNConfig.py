from enum import Enum


class DQNConfig(Enum):

    DEFAULT = (10000, 100, 0.99, 0.1, [32, 32], 1)

    def __init__(self, total_timesteps, max_episode_steps, gamma, learning_rate, net_arch, verbose):

        self.total_timesteps = total_timesteps
        self.max_episode_steps = max_episode_steps
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.net_arch = net_arch
        self.verbose = verbose
