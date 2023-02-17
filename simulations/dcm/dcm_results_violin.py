import pandas as pd

from utils.charts import get_violinchart, get_scatterplot
from utils.misc import load_from_csv

# 300
# DBSCAN
db_data_300 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcm_cluster_statistics_300-backup.csv")
db_data_300.columns = ['Mean Number of Clusters', 'Mean Number of Samples per Clusters', 'Number of Outliers']
db_data_300 = db_data_300.assign(Algorithm='DBSCAN')

# KMeans
kmeans_data_300 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmk_cluster_statistics_300-backup.csv")
kmeans_data_300.columns = ['Mean Number of Clusters', 'Mean Number of Samples per Clusters', 'Number of Outliers']
kmeans_data_300 = kmeans_data_300.assign(Algorithm='KMeans')

# Birch
birch_data_300 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmb_cluster_statistics_300-backup.csv")
birch_data_300.columns = ['Mean Number of Clusters', 'Mean Number of Samples per Clusters', 'Number of Outliers']
birch_data_300 = birch_data_300.assign(Algorithm='Birch')

# result_300 = pd.concat([db_data_300, kmeans_data_300, birch_data_300])
result_300 = pd.concat([kmeans_data_300, birch_data_300])

# 600
# DBSCAN
db_data_600 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcm_cluster_statistics_600-backup.csv")
db_data_600.columns = ['Mean Number of Clusters', 'Mean Number of Samples per Clusters', 'Number of Outliers']
db_data_600 = db_data_600.assign(Algorithm='DBSCAN')

# KMeans
kmeans_data_600 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmk_cluster_statistics_600-backup.csv")
kmeans_data_600.columns = ['Mean Number of Clusters', 'Mean Number of Samples per Clusters', 'Number of Outliers']
kmeans_data_600 = kmeans_data_600.assign(Algorithm='KMeans')

# Birch
birch_data_600 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmb_cluster_statistics_600-backup.csv")
birch_data_600.columns = ['Mean Number of Clusters', 'Mean Number of Samples per Clusters', 'Number of Outliers']
birch_data_600 = birch_data_600.assign(Algorithm='Birch')

# result_600 = pd.concat([db_data_600, kmeans_data_600, birch_data_600])
result_600 = pd.concat([kmeans_data_600, birch_data_600])

# 900
# DBSCAN
db_data_900 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcm_cluster_statistics_900-backup.csv")
db_data_900.columns = ['Mean Number of Clusters', 'Mean Number of Samples per Clusters', 'Number of Outliers']
db_data_900 = db_data_900.assign(Algorithm='DBSCAN')

# KMeans
kmeans_data_900 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmk_cluster_statistics_900-backup.csv")
kmeans_data_900.columns = ['Mean Number of Clusters', 'Mean Number of Samples per Clusters', 'Number of Outliers']
kmeans_data_900 = kmeans_data_900.assign(Algorithm='KMeans')

# Birch
birch_data_900 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmb_cluster_statistics_900-backup.csv")
birch_data_900.columns = ['Mean Number of Clusters', 'Mean Number of Samples per Clusters', 'Number of Outliers']
birch_data_900 = birch_data_900.assign(Algorithm='Birch')

# result_900 = pd.concat([db_data_900, kmeans_data_900, birch_data_900])
result_900 = pd.concat([kmeans_data_900, birch_data_900])

# Plot
# Violin plot
# get_violinchart(result_300, 300)
# get_violinchart(result_600, 600)
# get_violinchart(result_900, 900)


# Scatter plot
get_scatterplot(result_300, 300)
get_scatterplot(result_600, 600)
get_scatterplot(result_900, 900)
