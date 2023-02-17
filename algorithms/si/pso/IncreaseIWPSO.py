from algorithms.si import IncreaseIWPSOParticle
from algorithms.si import PSO


class IncreaseIWPSO(PSO):

    def __init__(self, data, population_size, max_steps, clustering_method, inertia_weight, cognitive_factor):
        super().__init__(data, max_steps, clustering_method)
        self.inertia_weight = []

        # Initialize the inertia weight list
        initial_inertia = inertia_weight[0]
        final_inertia = inertia_weight[1]
        for step in range(max_steps):
            current_inertia = initial_inertia + ((step/max_steps)*(final_inertia - initial_inertia))
            self.inertia_weight.append(current_inertia)

        # Create the PSO population
        for i in range(population_size):
            self.population.append(IncreaseIWPSOParticle(self.clustering_method, len(self.data), cognitive_factor))

    def search(self):
        print("Starting IIWPSO Engine with {} particles and {} iterations".format(len(self.population), self.max_steps))
        counter = 0
        while counter < self.max_steps:
            # Evaluate the entire population
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
