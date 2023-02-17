import sys
import gym
import pickle
from stable_baselines3.common.env_checker import check_env


# Get user density
user_density = int(sys.argv[1])

# Get number of BSs
n_bs = int(sys.argv[2])


global filehandler

try:
    slice_test_filename = "/Users/hugo/Desktop/PyCRE/rlm/data/slice_list_computed_{}_{}.obj".format(user_density, n_bs)
    filehandler = open(slice_test_filename, 'rb')
    slice_list = pickle.load(filehandler)

except IOError:
    # Load cluster list
    filename = "/Users/hugo/Desktop/PyCRE/iom/data/slice_list_{}_{}_75_pop_400.obj".format(user_density, n_bs)
    filehandler = open(filename, 'rb')
    slice_list = pickle.load(filehandler)

    # Saving the biggest slice for testing pourposes
    slice_list.sort(key=lambda x: len(x.cluster.bs_list), reverse=True)
    for network_slice in slice_list:
        network_slice.compute_selected_bs()

    selected_filename = "/Users/hugo/Desktop/PyCRE/rlm/data/slice_list_computed_{}_{}.obj".format(user_density, n_bs)
    filehandler = open(selected_filename, 'wb')
    pickle.dump(slice_list, filehandler)

finally:
    filehandler.close()

# Debug
print("Starting RLM with {} UEs/km2 and {} BSs/km2".format(user_density, n_bs))

for id_ in range(5):
    network_slice = slice_list[id_]
    env = gym.make("gym_pycre:pycre-v2", network_slice=network_slice)
    check_env(env)