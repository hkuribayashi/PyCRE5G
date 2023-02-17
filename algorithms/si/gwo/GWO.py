from algorithms.si import Wolf


class GWO:

    def __init__(self, cluster, max_steps, population_size, pareto_weight):
        self.max_steps = max_steps
        self.population = []
        self.cluster = cluster
        self.a = []
        self.global_evaluation = []
        self.best_evaluation = []
        self.pareto_weight = pareto_weight

        # Initialize the Grey Wolf population
        solution_size = len(self.cluster.bs_list)
        for idx in range(population_size):
            self.population.append(Wolf(solution_size))

        # Initialize a
        for step in range(max_steps):
            current_a = 2.0 * (1.0 - step/self.max_steps)
            self.a.append(current_a)

        # Calculate the fitness of each search agent
        self.evaluate()

        # Sort the wolf pack
        self.population.sort(key=lambda x: x.evaluation, reverse=True)

        # Select the leaders
        self.alpha = Wolf(solution_size)
        self.alpha.solution = self.population[0].solution
        self.alpha.evaluation = self.population[0].evaluation

        self.beta = Wolf(solution_size)
        self.beta.solution = self.population[1].solution
        self.beta.evaluation = self.population[1].evaluation

        self.delta = Wolf(solution_size)
        self.delta.solution = self.population[2].solution
        self.delta.evaluation = self.population[2].evaluation

        self.population.pop(0)
        self.population.pop(0)
        self.population.pop(0)

    def evaluate(self):
        # Initialize aux variable
        sum_temp = 0

        # Sum each each particle evaluation
        for p in self.population:
            p.evaluate(self.pareto_weight, self.cluster.bs_list)
            sum_temp += p.evaluation

        # Compute the mean evaluation
        return sum_temp/len(self.population)

    def search(self):
        counter = 0
        while counter < self.max_steps:

            for p in self.population:
                if p.evaluation > self.alpha.evaluation:
                    self.delta.evaluation = self.beta.evaluation
                    self.delta.solution = self.beta.solution.copy()
                    self.beta.evaluation = self.alpha.evaluation
                    self.beta.solution = self.alpha.solution.copy()
                    self.alpha.evaluation = p.evaluation
                    self.alpha.solution = p.solution.copy()

                if self.alpha.evaluation > p.evaluation > self.beta.evaluation:
                    self.delta.evaluation = self.beta.evaluation
                    self.delta.solution = self.beta.solution.copy()
                    self.beta.evaluation = p.evaluation
                    self.beta.solution = p.solution.copy()
                if p.evaluation > self.delta.evaluation:
                    self.delta.evaluation = p.evaluation
                    self.delta.solution = p.solution.copy()

            for p in self.population:
                p.update_position(self.alpha, self.beta, self.delta, self.a[counter])
                p.adjust_position()

            # Get global mean evaluation
            evaluation = self.evaluate()
            self.global_evaluation.append(evaluation)

            # Get the best evaluation
            best_evaluation = self.alpha.evaluation
            self.best_evaluation.append(best_evaluation)

            print('Iteration {} - Mean Evaluation: {} | Alpha: {}'.format(counter, evaluation, best_evaluation))

            counter += 1
