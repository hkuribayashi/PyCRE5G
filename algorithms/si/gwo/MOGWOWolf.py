import random

import numpy as np
from functools import total_ordering


class MOGWOWolf:
    def __init__(self, idx, solution_size):
        self.idx = idx
        flag = True
        while flag:
            self.solution = []
            for _ in range(solution_size):
                self.solution.append(random.uniform(0, 1))
            count = len([componente for componente in self.solution if componente >= 0.5])
            if count >= 1:
                flag = False

        self.evaluation_f1 = np.Inf
        self.evaluation_f2 = np.Inf
        self.solution_size = solution_size

    def __hash__(self):
        return self.idx

    def evaluate(self, bs_list):
        n_bs = len([componente for componente in self.solution if componente >= 0.5])
        n_rb = 0
        total_rb = 0
        for id_, bs in enumerate(bs_list):
            if self.solution[id_] > 0.5:
                n_rb += bs.resouce_blocks if bs.load == 0 else bs.resouce_blocks/bs.load
            total_rb += bs.resouce_blocks if bs.load == 0 else bs.resouce_blocks / bs.load
        self.evaluation_f1 = (-1) * (n_rb/total_rb)
        self.evaluation_f2 = n_bs/len(bs_list)

    def update_position(self, alpha, beta, delta, a):
        for idx, componente in enumerate(self.solution):
            # Alpha
            r1 = random.random()
            r2 = random.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            D_alpha = abs(C * alpha.solution[idx] - componente)
            x1 = alpha.solution[idx] - A * D_alpha

            # Beta
            r1 = random.random()
            r2 = random.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            D_beta = abs(C * beta.solution[idx] - componente)
            x2 = beta.solution[idx] - A * D_beta

            # Delta
            r1 = random.random()
            r2 = random.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            D_delta = abs(C * delta.solution[idx] - componente)
            x3 = delta.solution[idx] - A * D_delta

            # Compute the mean position
            new_position = (x1 + x2 + x3) / 3.0
            self.solution[idx] = new_position

    def __eq__(self, other):
        if isinstance(other, MOGWOWolf):
            return self.idx == other.idx
        return False

    def __str__(self):
        return "[MOGWOWolf id={}, evaluation_f1={}, evaluation_f2={}, solution={}]".format(self.idx, self.evaluation_f1, self.evaluation_f2, self.solution)

    @total_ordering
    def __lt__(self, other):
        return self.evaluation_f1 < other.evaluation_f1 and self.evaluation_f2 < other.evaluation_f2

    def __le__(self, other):
        return self.evaluation_f1 <= other.evaluation_f1 and self.evaluation_f2 <= other.evaluation_f2
