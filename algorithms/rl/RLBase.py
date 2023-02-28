from abc import ABC


class RLBase(ABC):
    def __init__(self, teste):
        self.teste = teste

    @staticmethod
    def learn():
        pass


class DQN(RLBase):

    def learn(self):
        a = 2
        print(a)
