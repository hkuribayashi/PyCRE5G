import sys
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

from config.network import Network


# Get user density
user_density = int(sys.argv[1])

# Get number of BSs
n_bs = int(sys.argv[2])

# Load cluster list object
filename = "/Users/hugo/Desktop/PyCRE/iom/data/cluster_list_{}_{}.obj".format(user_density, n_bs)
filehandler = open(filename, 'rb')
cluster_list = pickle.load(filehandler)

print("Number of Clusters: {}".format(len(cluster_list)))
number_of_ues = []
target_ues_75 = []
target_ues_50 = []

for id_, cluster in enumerate(cluster_list):
    # Cluster
    # Print Cluster Size (Number of UEs)
    n_ues = len(cluster.ue_list)
    print("Number of UEs: {}".format(n_ues))
    number_of_ues.append(n_ues)

    if cluster.evaluation['satisfaction'] < 75.0:
        target_ues_75.append(n_ues)

    if cluster.evaluation['satisfaction'] < 50.0:
        target_ues_50.append(n_ues)

df = DataFrame(number_of_ues, columns=['Number of UEs'])
df_75 = DataFrame(target_ues_75, columns=['Number of UEs'])
df_50 = DataFrame(target_ues_50, columns=['Number of UEs'])

if user_density == 300:
    bin_range = range(1, 55, 3)
elif user_density == 600:
    bin_range = range(1, 100, 3)
else:
    bin_range = range(1, 150, 3)

fig, ax = plt.subplots()
sns.histplot(data=df, x="Number of UEs", kde=False, color='#b1bfe1', label="Total Clusters", bins=bin_range, zorder=10)
sns.histplot(data=df_75, x="Number of UEs", kde=False, color="#e8b4a0", label="Total Outage Clusters", bins=bin_range, zorder=10)
plt.ylabel("Frequency")
plt.grid(linestyle=':', zorder=-1)
plt.legend(loc='best')
plt.savefig('{}{}'.format("/Users/hugo/Desktop/PyCRE/iom/images/", "iom_hist_75_{}_{}.eps".format(user_density, n_bs)), dpi=Network.DEFAULT.image_resolution, bbox_inches='tight')
plt.close()

plt.figure()
_, _ = plt.subplots()
sns.histplot(data=df, x="Number of UEs", kde=False, color='#b1bfe1', label="Total Clusters", bins=bin_range, zorder=10)
sns.histplot(data=df_50, x="Number of UEs", kde=False, color="#e8b4a0", label="Total Outage Clusters", bins=bin_range, zorder=10)
plt.ylabel("Frequency")
plt.grid(linestyle=':', zorder=-1)
plt.legend(loc='best')
plt.savefig('{}{}'.format("/Users/hugo/Desktop/PyCRE/iom/images/", "iom_hist_50_{}_{}.eps".format(user_density, n_bs)), dpi=Network.DEFAULT.image_resolution, bbox_inches='tight')
plt.close()
