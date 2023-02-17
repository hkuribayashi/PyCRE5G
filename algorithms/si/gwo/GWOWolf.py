import random
import numpy as np


class Wolf:
    def __init__(self, solution_size):
        self.solution = []
        for _ in range(solution_size):
            self.solution.append(random.uniform(0, 1))
        self.evaluation = np.NINF

    def evaluate(self, pareto_weight, bs_list):
        n_bs = len([componente for componente in self.solution if componente >= 0.5])
        n_rb = 0
        total_rb = 0
        for id_, bs in enumerate(bs_list):
            if self.solution[id_] >= 0.5:
                n_rb += bs.resouce_blocks if bs.load == 0 else bs.resouce_blocks/bs.load
            total_rb += bs.resouce_blocks if bs.load == 0 else bs.resouce_blocks / bs.load
        if len(bs_list) == 0:
            print("================>")
        self.evaluation = pareto_weight * (n_rb/total_rb) + (-1) * (1 - pareto_weight) * (n_bs/len(bs_list))

    def __validate_position(self):
        flag = True
        for id_, solution in enumerate(self.solution):
            if solution >= 0.5:
                flag = False
                break
        return flag

    def adjust_position(self):
        if self.__validate_position():
            max_position = self.solution.index(max(self.solution))
            self.solution[max_position] = 0.5

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
            self.solution[idx] = (x1 + x2 + x3)/3.0
