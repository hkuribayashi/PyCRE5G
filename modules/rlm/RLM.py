from algorithms.rl.qlearning.Environment import Environment
from config.DQNConfig import DQNConfig
from modules.rlm.ReinforcementLearningMethod import ReinforcementLearningMethod
from algorithms.rl.dqn.DQN import DQN
from algorithms.rl.qlearning.QLSingleAgent import QLSigleAgent


class RLM:

    def __init__(self, id_, rl_method, network_slice, config, model=None):
        if rl_method == ReinforcementLearningMethod.QLEARNING:
            env = Environment(network_slice)
            self.rl_engine = QLSigleAgent(env, config)
        elif rl_method == ReinforcementLearningMethod.DQN:
            self.rl_engine = DQN(id_, network_slice, config, model)
        self.evaluation = None

    def learn(self):
        return self.rl_engine.learn()

    def run(self):
        self.rl_engine.run()
        self.evaluation = self.rl_engine.evaluation
