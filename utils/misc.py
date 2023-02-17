import os
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, Birch


def get_pathloss(type_, distance):
    if type_ == 'MBS':
        pathloss = 128.0 + (37.6 * np.log10(max(distance, 35.0) / 1000.0))
    else:
        pathloss = 140.7 + (36.7 * np.log10((max(distance, 10.0) / 1000.0)))

    return pathloss


def get_distance(point_a, point_b):
    return ((point_b.x - point_a.x) ** 2 + (point_b.y - point_a.y) ** 2 + (point_b.z - point_a.z) ** 2) ** 0.5


def get_efficiency(sinr):
    if sinr >= 17.6:
        efficiency = 5.55
    elif sinr >= 16.8:
        efficiency = 5.12
    elif sinr >= 15.6:
        efficiency = 4.52
    elif sinr >= 13.8:
        efficiency = 3.9
    elif sinr >= 13.0:
        efficiency = 3.32
    elif sinr >= 11.8:
        efficiency = 2.73
    elif sinr >= 11.4:
        efficiency = 2.41
    elif sinr >= 10.0:
        efficiency = 1.91
    elif sinr >= 6.6:
        efficiency = 1.48
    elif sinr >= 3.0:
        efficiency = 1.18
    elif sinr >= 1.0:
        efficiency = 0.88
    elif sinr >= -1.0:
        efficiency = 0.6
    elif sinr >= -2.6:
        efficiency = 0.38
    elif sinr >= -4.0:
        efficiency = 0.23
    else:
        efficiency = 5.55

    return efficiency


def save_to_csv(data, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def load_from_csv(path, filename):
    data = pd.read_csv('{}{}'.format(path, filename), header=None, delimiter=',', sep=',')
    return data


def load_from_csv_number(path, filename, number):
    full_path = os.path.join(path, filename)
    cols = np.arange(number)
    data = pd.read_csv(full_path, header=None, delimiter=',', sep=',', names=cols)
    return data


def consolidate_results(path, filename):
    data = pd.read_csv('{}{}'.format(path, filename), header=None, delimiter=',', sep=',')
    return data.satisfaction_growth(axis=0)


def get_k_closest_bs(ue, bs_list):
    min_distance = 100000000.0
    idx_min = -1
    for idx, bs in enumerate(bs_list):
        distance = get_distance(ue.point, bs.point)
        if distance < min_distance and bs.type == 'SBS':
            min_distance = distance
            idx_min = idx
    closest_bs = [bs_list[idx_min]]
    return closest_bs


def get_statistics_dbscan(epsilon, min_samples, data):
    db_cluster = DBSCAN(eps=epsilon, min_samples=min_samples).fit(data)
    labels = db_cluster.labels_
    n_noise_ = list(labels).count(-1)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters_ >= 1:
        mean_cluster_size = (len(data) - n_noise_) / n_clusters_
    else:
        mean_cluster_size = 0

    return n_clusters_, mean_cluster_size, n_noise_


def get_statistics_kmeans(k, data):
    new_k = get_int(k, len(data))
    db_cluster = KMeans(n_clusters=new_k, random_state=170).fit(data)
    labels = db_cluster.labels_
    n_noise_ = list(labels).count(-1)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters_ >= 1:
        mean_cluster_size = (len(data) - n_noise_) / n_clusters_
    else:
        mean_cluster_size = 0

    return n_clusters_, mean_cluster_size, n_noise_


def get_statistics_birch(k, data):
    new_k = get_int(k, len(data))
    db_cluster = Birch(n_clusters=new_k).fit(data)
    labels = db_cluster.labels_
    n_noise_ = list(labels).count(-1)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters_ >= 1:
        mean_cluster_size = (len(data) - n_noise_) / n_clusters_
    else:
        mean_cluster_size = 0

    return n_clusters_, mean_cluster_size, n_noise_


def get_statistics_gaussian(k, data):
    new_k = get_int(k, len(data))
    db_cluster = Birch(n_clusters=new_k).fit(data)
    labels = db_cluster.labels_
    n_noise_ = list(labels).count(-1)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters_ >= 1:
        mean_cluster_size = (len(data) - n_noise_) / n_clusters_
    else:
        mean_cluster_size = 0

    return n_clusters_, mean_cluster_size, n_noise_


def get_ippp(simulation_area, lambda0, thinning_probability=0.4):
    side_length = np.sqrt(simulation_area)

    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    xDelta = x_max - x_min
    yDelta = y_max - y_min

    # Simulate a Poisson point process
    numbPoints = np.random.poisson(lambda0)  # Poisson number of points
    xx = np.random.uniform(0, xDelta, (numbPoints, 1)) + x_min  # x coordinates of Poisson points
    yy = np.random.uniform(0, yDelta, (numbPoints, 1)) + y_min  # y coordinates of Poisson points

    # Generate Bernoulli variables (ie coin flips) for thinning
    # points to be thinned
    booleThinned = np.random.uniform(0, 1, (numbPoints, 1)) > thinning_probability
    # points to be retained
    booleRetained = ~booleThinned

    # x/y locations of retained points
    xxRetained = xx[booleRetained] * side_length
    yyRetained = yy[booleRetained] * side_length

    return xxRetained, yyRetained


def get_int(k_number, data_size):
    max_ = int(0.1 * data_size)
    result = int(k_number)
    if k_number < 3:
        result = 3
    elif k_number > max_:
        result = max_
    elif k_number - result >= 0.5:
        result += 1
    return result


def get_int_2(branching_factor):

    if branching_factor < 10:
        result = 10
    elif branching_factor > 50:
        result = 50
    else:
        result = int(branching_factor)
    return result


def get_pareto_evaluation(slice, solution):
    bs_list = slice.cluster.bs_list
    n_bs = len([componente for componente in solution if componente >= 0.5])
    n_rb = 0
    total_rb = 0
    for id_, bs in enumerate(bs_list):
        if solution[id_] > 0.5:
            n_rb += bs.resouce_blocks if bs.load == 0 else bs.resouce_blocks/bs.load
        total_rb += bs.resouce_blocks if bs.load == 0 else bs.resouce_blocks / bs.load
    f1 =  (n_rb/total_rb)
    f2 = (-1) * n_bs/len(bs_list)

    return f1, f2


def get_mean_load(bs_list):
    mean = 0
    for bs in bs_list:
        mean += (bs.load / bs.max_load)
    result = (mean/len(bs_list)) * 100
    return result if result > 0 else 0.01
