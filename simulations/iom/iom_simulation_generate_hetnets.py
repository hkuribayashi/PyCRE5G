import pickle
import sys

from config.network import Network
from network.hetnet import HetNet


# Get traffic level
user_density = int(sys.argv[1])

# Get number of Hetnets
n_hetnets = int(sys.argv[2])

# Get number of BSs
n_bs = int(sys.argv[3])

# Path
path = sys.argv[4]

# Debug
print("User Density: {} UEs/km2".format(user_density))

counter = 0

while counter < n_hetnets:
    print("Current Hetnet: {}".format(counter))

    # Instantiate a HetNet
    h = HetNet(Network.DEFAULT)
    h.populate_bs(n_bs)

    # Run the HetNet
    h.run(user_density)

    filename = "{}hetnet_{}_{}_{}.obj".format(path, user_density, n_bs, counter)
    file = open(filename, 'wb')

    pickle.dump(h, file)
    file.close()

    counter += 1
