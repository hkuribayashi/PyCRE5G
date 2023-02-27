from config.DQNConfig import DQNConfig
from algorithms.rl.dqn.DQN import DQN


class RLM:

    def __init__(self, id_, network_slice, rl_method=None, config=None):

        if rl_method is None:
            if config is None:
                self.rl_engine = DQN(id_, network_slice, DQNConfig.DEFAULT)
            else:
                self.rl_engine = DQN(id_, network_slice, config)
        self.evaluation = None

    def learn(self):
        return self.rl_engine.learn()

    def run(self):
        self.rl_engine.run()
        self.evaluation = self.rl_engine.evaluation
