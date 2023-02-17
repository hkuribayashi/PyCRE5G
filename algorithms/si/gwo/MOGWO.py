import copy

from algorithms.si import MOGWOSegmentController
from algorithms.si import MOGWOWolf


def find_nondominated_solutions(populacao):
    non_dominated_list = [copy.deepcopy(populacao[0])]

    for p in populacao:
        non_dominated_list = list(set(non_dominated_list))
        list_del = []
        flag = False
        for non_dominated in non_dominated_list:
            if p <= non_dominated:
                flag = True
                list_del.append(non_dominated)
            elif p.evaluation_f1 <= non_dominated.evaluation_f1 or p.evaluation_f2 <= non_dominated.evaluation_f2:
                flag = True
            else:
                flag = False
        for element in list_del:
            non_dominated_list.remove(element)
        if flag:
            non_dominated_list.append(copy.deepcopy(p))

    return list(set(non_dominated_list))


class MOGWO:
    def __init__(self, cluster, max_steps, population_size, n_segments):
        self.cluster = cluster
        self.max_steps = max_steps
        self.population = []
        self.a = []
        self.archive = None

        # Initialize the Grey Wolf population
        solution_size = len(self.cluster.bs_list)

        # Initialize the Grey Wolf population
        for idx in range(population_size):
            self.population.append(MOGWOWolf(idx, solution_size))

        # Initialize a
        for step in range(max_steps):
            current_a = 2.0 * (1.0 - step / self.max_steps)
            self.a.append(current_a)

        # Calculate the fitness of each search agent
        self.evaluate()

        # Find the non-dominated solutions
        non_dominatead_solutions_set = []
        for seg in range(n_segments):
            population_ = []
            for idx in range(2):
                p_ = MOGWOWolf(idx, solution_size)
                p_.evaluate(self.cluster.bs_list)
                population_.append(p_)
            non_dominatead_solutions = find_nondominated_solutions(population_)
            non_dominatead_solutions_set.append(non_dominatead_solutions)

        # Initialize the archive with the non-dominated solutions
        self.segment_controller = MOGWOSegmentController(n_segments, non_dominatead_solutions_set)

        # Select the alpha leader
        self.alpha, alpha_segment_id = self.segment_controller.select_leader()

        # Select the beta leader
        self.beta, beta_segment_id = self.segment_controller.select_leader()

        # Select the delta leader
        self.delta, delta_segment_id = self.segment_controller.select_leader()

        # Add back alpha, beta and delta to the archive
        self.segment_controller.add_leader(alpha_segment_id, self.alpha)
        self.segment_controller.add_leader(beta_segment_id, self.beta)
        self.segment_controller.add_leader(delta_segment_id, self.delta)

    def evaluate(self):
        for p in self.population:
            p.evaluate(self.cluster.bs_list)

    def search(self):
        counter = 0
        while counter < self.max_steps:

            for p in self.population:
                p.update_position(self.alpha, self.beta, self.delta, self.a[counter])

            # Calculate the objective values of all search agents
            self.evaluate()

            # Find the non-dominated solutions
            non_dominated_list = find_nondominated_solutions(self.population)

            # Update the archive with respect to the obtained non-dominated solutions
            self.segment_controller.update(non_dominated_list)

            # Select the alpha leader
            self.alpha, alpha_segment_id = self.segment_controller.select_leader()

            # Select the beta leader
            self.beta, beta_segment_id = self.segment_controller.select_leader()

            # Select the delta leader
            self.delta, delta_segment_id = self.segment_controller.select_leader()

            # Add back alpha, beta and delta to the archive
            self.segment_controller.add_leader(alpha_segment_id, self.alpha)
            self.segment_controller.add_leader(beta_segment_id, self.beta)
            self.segment_controller.add_leader(delta_segment_id, self.delta)

            print("Step: {} | Number of solutions in the archive: {}".format(counter, self.segment_controller.get_archive_size()))

            counter += 1

        self.archive = self.segment_controller.get_archived_solutions()
