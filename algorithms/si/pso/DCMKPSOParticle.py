import random

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from utils.misc import get_int


class DCMKPSOParticle():

    def __init__(self, clustering_method, data_size, cognitive_factor):
        self.k = random.uniform(3, int(0.1 * data_size))
        self.best_k = self.k
        self.evaluation = 10.0
        self.data_size = data_size
        self.clustering_method = clustering_method
        self.c1 = cognitive_factor[0]
        self.c2 = cognitive_factor[1]

    def evaluate(self, data):
        n_clusters = get_int(self.k, len(data))
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, random_state=100, algorithm='full').fit(data)

        # Get the cluster's labels and total number of clusters
        labels = kmeans.labels_

        current_evaluation = davies_bouldin_score(data, labels)
        current_evaluation += (-1) * silhouette_score(data, labels)

        # Updates the evaluation variables
        if current_evaluation < self.evaluation:
            self.evaluation = current_evaluation
            self.best_k = self.k

    def update_position(self, g_best, inertia_weight):
        phi1 = random.random()
        phi2 = random.random()

        # Update k velocity
        velocity_k = (self.best_k - self.k) * phi1 * self.c1 + \
                           (g_best.best_k - self.k) * phi2 * self.c2

        # Update k position
        self.k = self.k + (inertia_weight * velocity_k)
