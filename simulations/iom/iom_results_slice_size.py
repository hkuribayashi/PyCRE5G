import sys
import pickle
import numpy as np


# Get user density
from utils.charts import get_bar_chart

user_density = int(sys.argv[1])

# Get number of BSs
n_bs = int(sys.argv[2])

# Get path
path = sys.argv[3]

# Load cluster list object
filename = "/Users/hugo/Desktop/PyCRE/iom/data/slice_list_{}_{}_75_pop_400.obj".format(user_density, n_bs)
filehandler = open(filename, 'rb')
slice_list = pickle.load(filehandler)

mean_n_bs = np.zeros(9)
mean_load_per_weight = np.zeros(9)

for id_, slice_ in enumerate(slice_list):
    total_bs = len(slice_.cluster.bs_list)
    mean_n_bs_per_weight = np.zeros(9)

    for i, (key, value) in enumerate(slice_.pareto_solutions.items()):
        solution = slice_.pareto_solutions[key]
        mean_n_bs_per_weight[i] += len([componente for componente in solution if componente >= 0.5])

        n_rb = 0
        total_rb = 0
        for id2_, bs in enumerate(slice_.cluster.bs_list):
            if solution[id2_] >= 0.5:
                n_rb += bs.load
            total_rb += bs.max_load
        mean_load_per_weight[i] += n_rb/total_rb

    mean_n_bs_per_weight = mean_n_bs_per_weight/total_bs
    mean_n_bs += mean_n_bs_per_weight

# Compute percentage
mean_n_bs = (mean_n_bs/len(slice_list)) * 100
mean_load_per_weight = (mean_load_per_weight/len(slice_list)) * 100

get_bar_chart(mean_n_bs, mean_load_per_weight, path, "iom_gwo_{}_{}.eps".format(user_density, n_bs))
