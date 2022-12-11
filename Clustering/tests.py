from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import numpy as np

# loading the dataset
X, y = datasets.load_iris("Clustering/Datasets/a.csv")

# K-Means
kmeans = KMeans(n_clusters=3, random_state=1).fit(X)

# we store the cluster labels
labels = kmeans.labels_
print(X)

print(metrics.calinski_harabasz_score(X, labels))
