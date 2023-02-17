import pandas as pd

from utils.charts import get_evaluation_evolution


weight_list = [0.9, 0.5, 0.1]

# CVS Path
path = "/Users/hugo/Desktop/PyCRE/iom/csv/"

chart_data = {}
for weight in weight_list:

    # 300/200 - GWO
    data_per_weight = []
    for id_ in range(0, 199):
        csv_filename = 'iom_300_cluster_mean_evolution_{}_75_GWO_{}_pop_400.csv'.format(weight, id_)
        data = pd.read_csv('{}{}'.format(path, csv_filename), header=None, delimiter=',', sep=',')
        data_per_weight.append(data)
    chart_data["$\lambda_{u}=300,\lambda_{2}=200$"] = pd.concat(data_per_weight).mean(axis=0)

    # 600/400 - GWO
    data_per_weight = []
    for id_ in range(0, 152):
        csv_filename = 'iom_600_cluster_mean_evolution_{}_75_GWO_{}_pop_400.csv'.format(weight, id_)
        data = pd.read_csv('{}{}'.format(path, csv_filename), header=None, delimiter=',', sep=',')
        data_per_weight.append(data)
    chart_data["$\lambda_{u}=600,\lambda_{2}=400$"] = pd.concat(data_per_weight).mean(axis=0)

    # 900/800 - GWO
    data_per_weight = []
    for id_ in range(0, 26):
        csv_filename = 'iom_900_cluster_mean_evolution_{}_75_GWO_{}_pop_400.csv'.format(weight, id_)
        data = pd.read_csv('{}{}'.format(path, csv_filename), header=None, delimiter=',', sep=',')
        data_per_weight.append(data)
    chart_data["$\lambda_{u}=900,\lambda_{2}=800$"] = pd.concat(data_per_weight).mean(axis=0)

    if weight == 0.1:
        ylim = (-0.26, -0.16)
    else:
        ylim = None

    # Plot
    get_evaluation_evolution(chart_data, 'iom_gwo_evolution_{}.eps'.format(weight), '-*', (-3, 305), ylim, False)

    # 300/200 - Alpha GWO
    data_per_weight = []
    for id_ in range(0, 199):
        csv_filename = 'iom_300_cluster_mean_evolution_{}_75_GWO_alpha_{}_pop_400.csv'.format(weight, id_)
        data = pd.read_csv('{}{}'.format(path, csv_filename), header=None, delimiter=',', sep=',')
        data_per_weight.append(data)
    chart_data["$\lambda_{u}=300,\lambda_{2}=200$"] = pd.concat(data_per_weight).mean(axis=0)

    # 600/400 - Alpha GWO
    data_per_weight = []
    for id_ in range(0, 152):
        csv_filename = 'iom_600_cluster_mean_evolution_{}_75_GWO_alpha_{}_pop_400.csv'.format(weight, id_)
        data = pd.read_csv('{}{}'.format(path, csv_filename), header=None, delimiter=',', sep=',')
        data_per_weight.append(data)
    chart_data["$\lambda_{u}=600,\lambda_{2}=400$"] = pd.concat(data_per_weight).mean(axis=0)

    # 900/800 - Alpha GWO
    data_per_weight = []
    for id_ in range(0, 26):
        csv_filename = 'iom_900_cluster_mean_evolution_{}_75_GWO_alpha_{}_pop_400.csv'.format(weight, id_)
        data = pd.read_csv('{}{}'.format(path, csv_filename), header=None, delimiter=',', sep=',')
        data_per_weight.append(data)
    chart_data["$\lambda_{u}=900,\lambda_{2}=800$"] = pd.concat(data_per_weight).mean(axis=0)

    if weight == 0.5:
        ylim = (0.065, 0.122)
    else:
        ylim = None

    # Plot
    get_evaluation_evolution(chart_data, 'iom_gwo_evolution_alpha_{}.eps'.format(weight), '-*', xlim=(-3, 102), ylim=ylim, alpha=True)
