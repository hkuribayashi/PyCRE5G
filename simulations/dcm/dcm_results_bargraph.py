from utils.charts import get_barchart
from utils.misc import load_from_csv

path = "/Users/hugo/Desktop/PyCRE/dcm/images/"

# DBSCAN
db_data_300 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcm_cluster_statistics_300-backup.csv")
db_data_600 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcm_cluster_statistics_600-backup.csv")
db_data_900 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcm_cluster_statistics_900-backup.csv")

db_data_300.columns = ['Number of Clusters', 'Number of Samples per Clusters', 'Number of Outliers']
db_data_600.columns = ['Number of Clusters', 'Number of Samples per Clusters', 'Number of Outliers']
db_data_900.columns = ['Number of Clusters', 'Number of Samples per Clusters', 'Number of Outliers']
dbscan_data = [db_data_300, db_data_600, db_data_900]

# K-Means
kmeans_data_300 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmk_cluster_statistics_300-backup.csv")
kmeans_data_600 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmk_cluster_statistics_600-backup.csv")
kmeans_data_900 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmk_cluster_statistics_900-backup.csv")

kmeans_data_300.columns = ['Number of Clusters', 'Number of Samples per Clusters', 'Number of Outliers']
kmeans_data_600.columns = ['Number of Clusters', 'Number of Samples per Clusters', 'Number of Outliers']
kmeans_data_900.columns = ['Number of Clusters', 'Number of Samples per Clusters', 'Number of Outliers']
kmeans_data = [kmeans_data_300, kmeans_data_600, kmeans_data_900]

# Birch
birch_data_300 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmb_cluster_statistics_300-backup.csv")
birch_data_600 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmb_cluster_statistics_600-backup.csv")
birch_data_900 = load_from_csv("/Users/hugo/Desktop/PyCRE/dcm/csv/", "dcmb_cluster_statistics_900-backup.csv")

birch_data_300.columns = ['Number of Clusters', 'Number of Samples per Clusters', 'Number of Outliers']
birch_data_600.columns = ['Number of Clusters', 'Number of Samples per Clusters', 'Number of Outliers']
birch_data_900.columns = ['Number of Clusters', 'Number of Samples per Clusters', 'Number of Outliers']
birch_data = [birch_data_300, birch_data_600, birch_data_900]

data = {'DBSCAN': dbscan_data, 'KMeans': kmeans_data, 'Birch': birch_data}

get_barchart(data, path)
