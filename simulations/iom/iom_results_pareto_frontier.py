import sys
import pickle
import numpy as np
from pandas import DataFrame

from utils.charts import get_pareto_frontier
from utils.misc import get_pareto_evaluation


# Get user density
user_density = int(sys.argv[1])

# Get number of BSs
n_bs = int(sys.argv[2])

# Get path
# path = sys.argv[3]

# Load cluster list object
filename = "/Users/hugo/Desktop/PyCRE/iom/data/slice_list_{}_{}_75_pop_400.obj".format(user_density, n_bs)
filehandler = open(filename, 'rb')
slice_list = pickle.load(filehandler)

# Load cluster list object
filename = "/Users/hugo/Desktop/PyCRE/iom/data/slice_list_{}_{}_75_pop_400_mogwo.obj".format(user_density, n_bs)
filehandler = open(filename, 'rb')
mo_slice_list = pickle.load(filehandler)

evaluation_f1 = np.zeros(9)
evaluation_f2 = np.zeros(9)

for slice_ in slice_list:

    for i, (key, value) in enumerate(slice_.pareto_solutions.items()):
        f1, f2 = get_pareto_evaluation(slice_, value)
        evaluation_f1[i] += f1
        evaluation_f2[i] += f2

all_points = []
for id_slice in range(20):
    mo_slice_ = mo_slice_list[id_slice]
    for d_ in mo_slice_.pareto_solutions:
        if d_.evaluation_f1 != 0 and d_.evaluation_f2 != 0:
            point = [(-1) * d_.evaluation_f1, (-1) * d_.evaluation_f2]
            all_points.append(point)

mogwo_df = DataFrame(all_points, columns=['x', 'y'])

path = "/Users/hugo/Desktop/PyCRE/iom/images/"
img_filename = "iom_pareto_frontier_{}_{}.eps".format(user_density, n_bs)
get_pareto_frontier(evaluation_f1 / len(slice_list), evaluation_f2 / len(slice_list), mogwo_df, path, img_filename)
