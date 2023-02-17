import sys

from modules.dcm.PSOAlgorithm import PSOAlgorithm
from modules.dcm.DCM import DCM
from modules.dcm.ClusteringMethod import ClusteringMethod
from config.network import Network
from mobility.point import Point
from network.bs import BS
from network.hetnet import HetNet
from utils.misc import save_to_csv


traffic_level = {'10': 100, '60': 600, '100': 999}

# Get total simulations
simulations = int(sys.argv[1])

# Get total iterations
iterations = int(sys.argv[2])

# Get population size
population_size = int(sys.argv[3])

# Output Directory
path = sys.argv[4]

# Debug
print("Running DCM with DCMPSO: {} simulations, {} iterations and {} particles".format(simulations,
                                                                                       iterations,
                                                                                       population_size))

for key in traffic_level:

    # Debug
    print("Traffic Level: {}%".format(key))

    # Initialize the result variables
    mean_evolution = []
    gbest_evolution = []

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
        h.run(traffic_level[key])

        # Instantiate DC Module with DBSCAM algorithm
        dcm = DCM(ClusteringMethod.DBSCAN, PSOAlgorithm.DCMPSO, h, traffic_level[key])

        # Run DCM
        dcm.optimization_engine(population_size, iterations)

        # Collect the generated results
        mean_evolution.append(dcm.optimization_output['DCMPSO-{}'.format(population_size)])
        gbest_evolution.append(dcm.optimization_output['DCMPSO-{}-gbest'.format(population_size)])

    print("\n")

    save_to_csv(mean_evolution, path, "mean_evolution_{}_pop_{}_DCMPSO.csv".format(key, population_size))
    save_to_csv(gbest_evolution, path, "mean_evolution_{}_pop_{}_gbest_DCMPSO.csv".format(key, population_size))
