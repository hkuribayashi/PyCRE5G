import sys

from modules.dcm.PSOAlgorithm import PSOAlgorithm
from modules.dcm.DCM import DCM
from modules.dcm.ClusteringMethod import ClusteringMethod
from config.network import Network
from mobility.point import Point
from network.bs import BS
from network.hetnet import HetNet
from utils.misc import save_to_csv


# Get traffic level
user_density = int(sys.argv[1])

# Get total simulations
simulations = int(sys.argv[2])

# Get total iterations
iterations = int(sys.argv[3])

# Get population size
population_size = int(sys.argv[4])

# Output Directory
path = sys.argv[5]

# Debug
print("Running DCM with DCMBPSO: {} simulations, {} iterations and {} particles".format(simulations,
                                                                                       iterations,
                                                                                       population_size))

# Debug
print("User Density: {} UEs/km2".format(user_density))

total_results = []
# TODO: Incluir o número de repetições na configuração DEFAULT
for idx in range(simulations):
    # Current Step
    print("Simulation: {}".format(idx))

    # Instantiate a HetNet
    h = HetNet(Network.DEFAULT)

    # Deploy a MBS
    p0 = Point(0.0, 0.0, 35.0)
    mbs = BS(0, 'MBS', p0)

    # Add each BS in the HetNet
    h.add_bs(mbs)

    # Run the HetNet
    h.run(user_density)

    # Instantiate DC Module with DBSCAM algorithm
    dcm = DCM(ClusteringMethod.GAUSSIAN_MIX, PSOAlgorithm.DCMPSO, h, user_density)

    # Run DCM
    results = dcm.get_cluster_analysis(population_size, iterations)
    total_results.append(results)

# Save Results
save_to_csv(total_results, path, "dcmb_cluster_statistics_{}.csv".format(user_density))
