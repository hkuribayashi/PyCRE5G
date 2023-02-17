import numpy as np
import itertools
from collections import Counter

from sklearn.cluster import DBSCAN, KMeans, Birch
from modules.dcm.Cluster import Cluster
from modules.dcm.ClusteringMethod import ClusteringMethod
from modules.dcm.PSOAlgorithm import PSOAlgorithm
from algorithms.si import CoPSO
from algorithms.si import DCMBPSO
from algorithms.si import DCMKPSO
from algorithms.si import IncreaseIWPSO
from algorithms.si import StochasticIWPSO
from algorithms.si import DCMPSO
from utils.misc import get_k_closest_bs, get_statistics_dbscan, get_statistics_kmeans, get_statistics_birch, get_int, get_int_2


class DCM:
    def __init__(self, clustering_method, pso_algorithm, hetnet, user_density):
        self.method = clustering_method
        self.pso_algorithm = pso_algorithm
        self.data = []
        self.optimization_output = {}
        self.ue_list = hetnet.ue_list
        self.bs_list = hetnet.list_bs
        self.clusters = []
        self.priority_ues_weight = hetnet.env.priority_ues_weight
        self.ordinary_ues_weight = hetnet.env.ordinary_ues_weight
        self.outage_threshold = hetnet.env.outage_threshold
        self.user_density = user_density
        self.clustering_output = None

        # Extract UE position to a numpy array
        for ue in self.ue_list:
            self.data.append([ue.point.x, ue.point.y])
        self.data = np.array(self.data)

    def compute_clusters(self, population_size=150, max_steps=150, flag=True):
        self.__get_optimized_cluster(population_size, max_steps)
        self.__get_target_clusters()
        self.__get_evaluation_per_cluster(flag)
        self.__compute_bs_per_cluster()

    def __get_optimized_cluster(self, population_size, max_steps):
        if self.method == ClusteringMethod.DBSCAN:
            pso = DCMPSO(self.data, population_size, max_steps, self.method, [0.9, 0.6], [2.05, 2.05])
            pso.search()
            self.clustering_output = DBSCAN(min_samples=pso.g_best.best_min_samples, eps=pso.g_best.best_epsilon).fit(self.data)

        elif self.method == ClusteringMethod.KMEANS:
            pso = DCMKPSO(self.data, population_size, max_steps, self.method, [0.9, 0.6], [2.05, 2.05])
            pso.search()
            self.clustering_output = KMeans(n_clusters=pso.g_best.best_k, init='k-means++', max_iter=100, random_state=100, algorithm='full').fit(self.data)
        elif self.method == ClusteringMethod.BIRCH:
            pso = DCMBPSO(self.data, population_size, max_steps, self.method, [0.9, 0.6], [2.05, 2.05])
            pso.search()
            self.clustering_output = Birch(n_clusters=get_int(pso.g_best.best_k, len(self.data)), branching_factor=get_int_2(pso.g_best.best_branching_factor)).fit(self.data)

    def __get_target_clusters(self):
        labels = self.clustering_output.labels_

        counter = Counter(labels.tolist())
        grouped_labels = [[k, ]*v for k, v in counter.items()]
        label_ids = []
        for group in grouped_labels:
            if group[0] is not -1:
                label_ids.append(group[0])

        for label_id in label_ids:
            new_cluster = Cluster(label_id, self.priority_ues_weight, self.ordinary_ues_weight, self.outage_threshold)
            for idx, l_ in enumerate(labels.tolist()):
                if label_id == l_:
                    new_cluster.ue_list.append(self.ue_list[idx])
            self.clusters.append(new_cluster)

    def __get_evaluation_per_cluster(self, flag=True):
        for cluster in self.clusters:
            cluster.evaluate(flag)
        self.clusters = [cluster for cluster in self.clusters if cluster.cluster is True]

    def __compute_bs_per_cluster(self):
        # Compute the closest BSs per cluster
        for cluster in self.clusters:
            bs_set = set()
            for ue in cluster.ue_list:
                closest_bs = get_k_closest_bs(ue, self.bs_list)
                bs_set.add(closest_bs[0])
            cluster.bs_list = bs_set

        # Compute difference between the BS sets
        self.clusters.sort(key=lambda x: x.evaluation['total_priority_ues'], reverse=True)
        for a, b in itertools.combinations(self.clusters, 2):
            a.bs_list = a.bs_list.difference(b.bs_list)
            b.bs_list = b.bs_list.difference(a.bs_list)

        # Debug
        for cluster in self.clusters:
            cluster.bs_list = list(sorted(cluster.bs_list))

    def remove_bs(self, closest_bs, cluster_id):
        for cluster in self.clusters:
            if cluster.id != cluster_id and closest_bs in cluster.bs_list:
                if isinstance(cluster.bs_list, list) :
                    cluster.bs_list.remove(closest_bs)
                elif isinstance(cluster.bs_list, set):
                    cluster.bs_list.remove(closest_bs)

    def optimization_engine(self, population_size, max_steps=20):
        if self.pso_algorithm is PSOAlgorithm.DCMPSO:
            pso = DCMPSO(self.data, population_size, max_steps, self.method, [0.9, 0.6], [2.05, 2.05])
            pso.search()
            self.optimization_output = {'DCMPSO-{}'.format(population_size): pso.mean_evaluation_evolution,
                                        'DCMPSO-{}-gbest'.format(population_size): pso.gbest_evaluation_evolution}
        elif self.pso_algorithm is PSOAlgorithm.CoPSO:
            pso = CoPSO(self.data, population_size, max_steps, self.method, [2.05, 2.05])
            pso.search()
            self.optimization_output = {'CoPSO-{}'.format(population_size): pso.mean_evaluation_evolution,
                                        'CoPSO-{}-gbest'.format(population_size): pso.gbest_evaluation_evolution}
        elif self.pso_algorithm is PSOAlgorithm.IncreaseIWPSO:
            pso = IncreaseIWPSO(self.data, population_size, max_steps, self.method, [0.4, 0.9], [2.0, 2.0])
            pso.search()
            self.optimization_output = {'IIWPSO-{}'.format(population_size): pso.mean_evaluation_evolution,
                                        'IIWPSO-{}-gbest'.format(population_size): pso.gbest_evaluation_evolution}
        elif self.pso_algorithm is PSOAlgorithm.StochasticIWPSO:
            pso = StochasticIWPSO(self.data, population_size, max_steps, self.method, [0.5, 1.0], [2.05, 2.05])
            pso.search()
            self.optimization_output = {'SIWPSO-{}'.format(population_size): pso.mean_evaluation_evolution,
                                        'SIWPSO-{}-gbest'.format(population_size): pso.gbest_evaluation_evolution}

    def get_cluster_analysis(self, population_size=150, max_steps=30):
        n_clusters = None
        mean_cluster_size = None
        n_outliers = None
        if self.method is ClusteringMethod.DBSCAN:
            flag = True
            while flag:
                pso = DCMPSO(self.data, population_size, max_steps, self.method, [0.9, 0.6], [2.05, 2.05])
                pso.search()
                if pso.g_best.evaluation < 10.0:
                    n_clusters, mean_cluster_size, n_outliers = get_statistics_dbscan(pso.g_best.best_epsilon, pso.g_best.best_min_samples, self.data)
                    flag = False
        elif self.method is ClusteringMethod.KMEANS:
            flag = True
            while flag:
                pso = DCMKPSO(self.data, population_size, max_steps, self.method, [0.9, 0.6], [2.05, 2.05])
                pso.search()
                if pso.g_best.evaluation < 10.0:
                    n_clusters, mean_cluster_size, n_outliers = get_statistics_kmeans(pso.g_best.best_k,
                                                                                      self.data)
                    flag = False
        elif self.method is ClusteringMethod.BIRCH:
            flag = True
            while flag:
                pso = DCMBPSO(self.data, population_size, max_steps, self.method, [0.9, 0.6], [2.05, 2.05])
                pso.search()
                if pso.g_best.evaluation < 10.0:
                    n_clusters, mean_cluster_size, n_outliers = get_statistics_birch(pso.g_best.best_k,
                                                                                     self.data)
                    flag = False

        return [n_clusters, mean_cluster_size, n_outliers]
