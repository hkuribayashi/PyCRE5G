class Slice:

    def __init__(self, cluster, pareto_solutions):
        self.cluster = cluster
        self.pareto_solutions = pareto_solutions

    def compute_selected_bs(self, pareto_weight=0.5):
        self.selected_bs = []
        solution = self.pareto_solutions[pareto_weight]
        for id_, value in enumerate(solution):
            if value >= 0.5:
                self.selected_bs.append(self.cluster.bs_list[id_])
