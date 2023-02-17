import sys
import pickle


# Get user density
user_density = int(sys.argv[1])

# Get number of BSs
n_bs = int(sys.argv[2])

# Load cluster list object
filename = "/Users/hugo/Desktop/PyCRE/iom/data/cluster_list_{}_{}.obj".format(user_density, n_bs)
filehandler = open(filename, 'rb')
cluster_list = pickle.load(filehandler)

target_ues_75 = []
target_ues_50 = []

for cluster in cluster_list:

    # Get all target clusters
    if cluster.evaluation['satisfaction'] < 75.0:
        target_ues_75.append(cluster)

    if cluster.evaluation['satisfaction'] < 50.0:
        target_ues_50.append(cluster)

# Save target clusters
# 75
filename = "/Users/hugo/Desktop/PyCRE/iom/data/target_cluster_list_{}_{}_75.obj".format(user_density, n_bs)
file = open(filename, 'wb')
pickle.dump(target_ues_75, file)
file.close()

# 50
filename = "/Users/hugo/Desktop/PyCRE/iom/data/target_cluster_list_{}_{}_50.obj".format(user_density, n_bs)
file = open(filename, 'wb')
pickle.dump(target_ues_50, file)
file.close()
