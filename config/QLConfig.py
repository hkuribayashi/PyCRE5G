from enum import Enum


class QLConfig(Enum):

    DEFAULT = (100, 100, 0.1, 0.99, 1, 0.01, 0.01, 20, True)

    def __init__(self, episodes, max_steps_per_episode, learning_rate, discount_rate, max_exploration_rate,
                 min_exploration_rate, exploration_decay_rate, max_iter_per_state, verbose):

        self.num_episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode

        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        self.max_iter_per_state = max_iter_per_state
        self.verbose = verbose
