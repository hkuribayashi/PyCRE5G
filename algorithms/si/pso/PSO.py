from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from operator import attrgetter
import numpy as np


class PSO(ABC):

    def __init__(self, data, max_steps, clustering_method):
        self.population = list()
        self.g_best = None
        self.global_evaluation = np.Inf
        self.max_steps = max_steps
        self.data = StandardScaler().fit_transform(data)
        self.clustering_method = clustering_method
        self.mean_evaluation_evolution = []
        self.gbest_evaluation_evolution = []

    def evaluate(self):
        # Initialize aux variable
        sum_temp = 0

        # Sum each each particle evaluation
        for p in self.population:
            p.evaluate(self.data)
            sum_temp += p.evaluation

        # Compute the mean evaluation
        self.global_evaluation = sum_temp/len(self.population)

        # Get the best particle
        self.g_best = min(self.population, key=attrgetter('evaluation'))

    @abstractmethod
    def search(self):
        pass
