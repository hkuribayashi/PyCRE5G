import random
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.cluster import Birch
from sklearn.metrics import davies_bouldin_score, silhouette_score

from utils.misc import get_int, get_int_2


class DCMBPSOParticle():

    def __init__(self, clustering_method, data_size, cognitive_factor):
        self.k = random.uniform(3, int(0.1 * data_size))
        self.best_k = self.k
        self.branching_factor = random.randint(10, 50)
        self.best_branching_factor = self.branching_factor
        self.evaluation = 10.0
        self.data_size = data_size
        self.clustering_method = clustering_method
        self.c1 = cognitive_factor[0]
        self.c2 = cognitive_factor[1]

    @ignore_warnings(category=ConvergenceWarning)
    def evaluate(self, data):
        n_clusters = get_int(self.k, len(data))
        branching_factor = get_int_2(self.branching_factor)
        birch = Birch(n_clusters=n_clusters, branching_factor=branching_factor).fit(data)

        # Get the cluster's labels and total number of clusters
        labels = birch.labels_

        current_evaluation = davies_bouldin_score(data, labels)
        current_evaluation += (-1) * silhouette_score(data, labels)

        # Updates the evaluation variables
        if current_evaluation < self.evaluation:
            self.evaluation = current_evaluation
            self.best_k = self.k
            self.best_branching_factor = self.branching_factor

    def update_position(self, g_best, inertia_weight):
        phi1 = random.random()
        phi2 = random.random()

        # Update k velocity
        velocity_k = (self.best_k - self.k) * phi1 * self.c1 + \
                           (g_best.best_k - self.k) * phi2 * self.c2

        # Update k position
        self.k = self.k + (inertia_weight * velocity_k)

        # Update threshold velocity
        velocity_branching_factor = (self.best_branching_factor - self.branching_factor) * phi1 * self.c1 + (
                    g_best.best_branching_factor - self.branching_factor) * phi2 * self.c2

        # Update threshold position
        self.branching_factor = self.branching_factor + (inertia_weight * velocity_branching_factor)
