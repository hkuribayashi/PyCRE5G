import os
import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import DQN as DQN_
from stable_baselines3.common.monitor import Monitor

from config.GlobalConfig import GlobalConfig


class DQN:
    def __init__(self, id_, network_slice, config):
        # Set up the ID value
        self.id_ = id_

        # Retrieving configurations
        policy = dict(net_arch=config.net_arch)

        # Check log dir
        self.log_dir = os.path.join(GlobalConfig.DEFAULT.rlm_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Retrieving the total number of steps and the max number of steps per episode
        self.max_episode_steps = config.max_episode_steps
        self.total_timesteps = config.total_timesteps

        # Instancia o ambiente Gym PyCRE
        self.env = gym.make("gym_pycre:pycre-v0", network_slice=network_slice)

        # Cria um Monitor
        self.env = Monitor(TimeLimit(env=self.env, max_episode_steps=self.max_episode_steps),
                           filename=os.path.join(GlobalConfig.DEFAULT.rlm_path, "logs"))

        # Verifica se há um modelo treinado
        path = os.path.join(GlobalConfig.DEFAULT.base_path, "models", "dqn1.zip")
        if os.path.isfile(path):
            # Carrega o modelo treinado
            self.model = DQN_.load(path)
            self.trainning = False
        else:
            # Carrega o modelo treinado e registra a necessidade de treinamento
            self.model = DQN_("MlpPolicy",
                              self.env,
                              policy_kwargs=policy,
                              learning_rate=config.learning_rate,
                              verbose=config.verbose)
            self.trainning = True

        # Avaliacao de Resultados
        self.evaluation = dict()
        self.evaluation["satisfaction"] = list()
        self.evaluation["load"] = list()
        self.evaluation["satisfaction"].append(network_slice.evaluation)

    def learn(self):
        if self.trainning is True:
            self.model.learn(total_timesteps=self.total_timesteps)
            learning_results = {"episode_lengths": self.env.get_episode_lengths(),
                                "episode_rewards": self.env.get_episode_rewards()}
            self.model.save(os.path.join(GlobalConfig.DEFAULT.base_path, "models", "dqn{}.zip".format(self.id_)))
            return learning_results

    def run(self):
        obs = self.env.reset()
        for _ in range(100):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.evaluation["satisfaction"].append(info["satisfaction"])
            self.evaluation["load"].append(info["mean_load"])
            if done:
                obs = self.env.reset()
