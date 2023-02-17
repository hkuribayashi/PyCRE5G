import numpy as np

from algorithms.si import CoPSOParticle
from algorithms.si import PSO


class CoPSO(PSO):

    def __init__(self, data, population_size, max_steps, clustering_method, cognitive_factor):
        super().__init__(data, max_steps, clustering_method)

        # Create the PSO population
        for i in range(population_size):
            self.population.append(CoPSOParticle(self.clustering_method, len(self.data), cognitive_factor))

        # Constriction Factor Parameter
        c = cognitive_factor[0] + cognitive_factor[1]

        # Computing the constriction factor value
        self.constricion_factor = 2.0/(np.abs(2 - c - np.sqrt((c**2) - 4*c)))

    def search(self):
        print("Starting CoPSO Engine with {} particles and {} iterations".format(len(self.population), self.max_steps))

        counter = 0
        while counter < self.max_steps:
            # Evaluate the population
            self.evaluate()

            # Update the particles' position
            for p in self.population:
                p.update_position(self.g_best, self.constricion_factor)

            # Save current mean evaluation and gbest evaluation
            self.mean_evaluation_evolution.append(self.global_evaluation)
            self.gbest_evaluation_evolution.append(self.g_best.evaluation)

            print('Iteration {} - Mean Evaluation: {} | Gbest Evaluation: {}'.format(counter, self.global_evaluation,
                                                                                     self.g_best.evaluation))

            # Increase the iteration counter
            counter += 1
