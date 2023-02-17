from enum import Enum


class ClusteringMethod(Enum):

    DBSCAN = 1,
    KMEANS = 2,
    BIRCH = 3,
    GAUSSIAN_MIX = 4

