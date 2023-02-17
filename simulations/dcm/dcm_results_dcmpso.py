from config.network import Network
from utils.charts import get_evaluation_evolution
from utils.misc import consolidate_results


traffic_level = {'10': 100, '60': 600, '100': 999}
population_size = [50, 100, 200]
chart_data = {}

# Plot mean evalution evolution
global mean_data
for key in traffic_level:
    chart_data = {}
    for population in population_size:
        csv_filename = 'mean_evolution_{}_pop_{}_DCMPSO.csv'.format(key, population)
        try:
            mean_data = consolidate_results("/Users/hugo/Desktop/PyCRE/dcm/csv/", csv_filename)
        except FileNotFoundError:
            mean_data = []
        finally:
            chart_data[population] = mean_data
    get_evaluation_evolution(chart_data, 'mean_evolution_list_{}.eps'.format(key), '-*')
