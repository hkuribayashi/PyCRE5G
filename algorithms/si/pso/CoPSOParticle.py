import random

from algorithms.si import PSOParticle


class CoPSOParticle(PSOParticle):

    def __init__(self, clustering_method, data_size, cognitive_factor):
        super().__init__(clustering_method, data_size, cognitive_factor)

    def update_position(self, g_best, constriction_factor):
        phi1 = random.random()
        phi2 = random.random()

        # Update epsilon velocity
        velocity_epsilon = (self.best_epsilon - self.epsilon) * phi1 * self.c1 + \
                           (g_best.best_epsilon - self.epsilon) * phi2 * self.c2

        # Update epsilon position
        self.epsilon = constriction_factor * (self.epsilon + velocity_epsilon)

        # Epsilon Constraint
        if self.epsilon < 0:
            self.epsilon = 0.1

        # Update Min samples velocity
        velocity_min_samples = (self.best_min_samples - self.min_samples) * phi1 * self.c1 + \
                               (g_best.best_min_samples - self.min_samples) * phi2 * self.c2

        # Update Min samples position
        self.min_samples = int(constriction_factor * (self.min_samples + velocity_min_samples))

        # Min Samples Constraint
        if self.min_samples < 2:
            self.min_samples = 2
        elif self.min_samples > self.data_size:
            self.min_samples = self.data_size - 1
