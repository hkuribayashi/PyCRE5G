import sys
import pickle

from modules.iom.GWOAlgorithm import GWOAlgorithm
from modules.iom.IOM import IOM


# Get user density
user_density = int(sys.argv[1])

# Get number of BSs
n_bs = int(sys.argv[2])

# Get satisfaction level
satisfaction_level = sys.argv[3]

# Get population size
pop_size = int(sys.argv[4])

# Get the OS path to save cvs files
path = sys.argv[5]

# Load cluster list object
filename = "/Users/hugo/Desktop/PyCRE/iom/data/target_cluster_list_{}_{}_{}.obj".format(user_density, n_bs, satisfaction_level)
filehandler = open(filename, 'rb')
target_cluster_list = pickle.load(filehandler)
selected_cluster_list = []

# Remove clusters with no BSs
# 300: 15 - 25
print("Cluster List Size: {}".format(len(target_cluster_list)))
for cluster in target_cluster_list:
    if 15 < len(cluster.ue_list) < 25 and len(cluster.bs_list) > 1:
        selected_cluster_list.append(cluster)

slice_list = []

# Compute a network slice for each target cluster
print("Updated Cluster List Size: {}".format(len(selected_cluster_list)))
for id_ in range(0, 62):

    # New Cluster ID
    print("Cluster {}".format(id_))

    cluster = selected_cluster_list[id_]

    # Instantiate IO Module with MOGWO Algorithm
    iom = IOM(cluster, path=path)

    # Compute the network slice using MOGWO approach
    gwo_slice = iom.compute_network_slice(id_, user_density, GWOAlgorithm.MOGWO, satisfaction_level=satisfaction_level, pop_size=pop_size)

    # Save the network slice
    slice_list.append(gwo_slice)

# Save the slice list
filename = "/Users/hugo/Desktop/PyCRE/iom/data/slice_list_{}_{}_{}_pop_{}_mogwo.obj".format(user_density, n_bs, satisfaction_level, pop_size)
file = open(filename, 'wb')
pickle.dump(slice_list, file)
file.close()
