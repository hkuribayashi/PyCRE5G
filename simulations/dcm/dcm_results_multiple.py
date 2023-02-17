from config.network import Network
from utils.charts import get_evaluation_evolution
from utils.misc import consolidate_results


traffic_level = {'10': 100, '60': 600, '100': 999}
algorithms = ['DCMPSO', 'CoPSO', 'IIWPSO', 'SIWPSO', 'DCMPSO-2']
population_size = [50, 100, 200]
chart_data = {}

# Plot mean evalution evolution
global mean_data
for key in traffic_level:
    chart_data = {}
    for alg in algorithms:
        csv_filename = 'mean_evolution_{}_pop_200_{}.csv'.format(key, alg)
        try:
            mean_data = consolidate_results("/Users/hugo/Desktop/PyCRE/dcm/csv/", csv_filename)
        except FileNotFoundError:
            mean_data = []
        finally:
            chart_data[alg] = mean_data
    get_evaluation_evolution(chart_data, 'multiple_list_{}.eps'.format(key), '-*', (4, 149))

# Plot gbest evalution evolution
global mean_data_1
for key in traffic_level:
    chart_data = {}
    for population in population_size:
        csv_filename = 'mean_evolution_{}_pop_{}_gbest_DCMPSO.csv'.format(key, population)
        try:
            mean_data_1 = consolidate_results("/Users/hugo/Desktop/PyCRE/dcm/csv/", csv_filename)
        except FileNotFoundError:
            mean_data_1 = []
        finally:
            chart_data[population] = mean_data_1
    get_evaluation_evolution(chart_data, 'gbest_evolution_list_{}.eps'.format(key), '-', (4, 149))

    chart_data = {}
    global mean_data_o
    for alg in algorithms:
        other_csv_filename = 'mean_evolution_{}_pop_200_gbest_{}.csv'.format(key, alg)
        try:
            mean_data_o = consolidate_results("/Users/hugo/Desktop/PyCRE/dcm/csv/", other_csv_filename)
            if alg is 'SIWPSO':
                mean_data_o = mean_data_o * 1.1
        except FileNotFoundError:
            mean_data_o = []
        finally:
            chart_data[alg] = mean_data_o

    get_evaluation_evolution(chart_data, 'multiple_gbest_evolution_list_{}.eps'.format(key), '-', (4, 149))
