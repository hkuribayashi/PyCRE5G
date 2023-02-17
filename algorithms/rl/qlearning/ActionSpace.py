from random import randrange


class ActionSpace:

    def __init__(self, size):
        self.n = size

    def sample(self):
        return randrange(self.n)
