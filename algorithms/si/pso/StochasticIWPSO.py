import random

from algorithms.si import PSO
from algorithms.si import StochasticIWPSOParticle


class StochasticIWPSO(PSO):

    def __init__(self, data, population_size, max_steps, clustering_method, inertia_weight, cognitive_factor):
        super().__init__(data, max_steps, clustering_method)
        self.inertia_weight = []

        # Create the PSO population
        for i in range(population_size):
            self.population.append(StochasticIWPSOParticle(self.clustering_method, len(self.data), cognitive_factor))

        # Initialize the inertia weight list
        for step in range(max_steps):
            self.inertia_weight.append(random.uniform(inertia_weight[0], inertia_weight[1]))

    def search(self):
        counter = 0
        while counter < self.max_steps:
            # Evaluate the population
            self.evaluate()

            # Update the particles' position
            for p in self.population:
                p.update_position(self.g_best, self.inertia_weight[counter])

            # Save current mean evaluation and gbest evaluation
            self.mean_evaluation_evolution.append(self.global_evaluation)
            self.gbest_evaluation_evolution.append(self.g_best.evaluation)

            print('Iteration {} - Mean Evaluation: {} | Gbest Evaluation: {}'.format(counter, self.global_evaluation,
                                                                                     self.g_best.evaluation))

            counter += 1
