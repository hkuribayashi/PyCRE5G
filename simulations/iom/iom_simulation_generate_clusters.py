import pickle
import sys

from modules.dcm.PSOAlgorithm import PSOAlgorithm
from modules.dcm.DCM import DCM
from modules.dcm.ClusteringMethod import ClusteringMethod

# Get traffic level
user_density = int(sys.argv[1])

# Get number of BSs
n_bs = int(sys.argv[2])

# Get the range
start = int(sys.argv[3])
stop = int(sys.argv[4])

# Path
path = sys.argv[5]

# Debug
print("User Density: {} UEs/km2".format(user_density))
print("Number of BSs: {}".format(n_bs))

cluster_list = []

for id_ in range(start, stop):
    print("Current number of clusters found: {}".format(len(cluster_list)))
    print("Loading Hetnet {}:".format(id_))

    # Load hetnet
    filename = '/Users/hugo/Desktop/PyCRE/iom/data/hetnet_{}_{}_{}.obj'.format(user_density, n_bs, id_)
    filehandler = open(filename, 'rb')
    hetnet = pickle.load(filehandler)

    # Instantiate DCM and compute clusters
    dcm = DCM(ClusteringMethod.BIRCH, PSOAlgorithm.DCMPSO, hetnet, user_density)

    # Optional flag setted to False to ignore the user satisfaction thrshold per cluster
    dcm.compute_clusters(max_steps=60, flag=False)

    # Get target clusters
    for target_cluster in dcm.clusters:
        print("Cluster Size: {} | Number of BSs: {}".format(len(target_cluster.ue_list), len(target_cluster.bs_list)))
        cluster_list.append(target_cluster)

filename = "{}cluster_list_{}_{}.obj".format(path, user_density, n_bs)
file = open(filename, 'wb')

pickle.dump(cluster_list, file)
file.close()
