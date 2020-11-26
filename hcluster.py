# HIERARCHICAL CLUSTERING
# SAMANTHA HERNANDEZ
# A01701448

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# IMPORT CSV MALL DATASET
dataset = pd.read_csv("dataset.csv")
x = dataset.iloc[:, [3, 4]].values

# FIND # OF CLUSTERS (DENDOGRAM)
dendrogram = sch.dendrogram(sch.linkage(x, method = "ward"))
plt.title("DENDOGRAM")
plt.xlabel("CLIENTS")
plt.ylabel("EUCLIDEAN DISTANCE")
plt.show()

# CLUSTER SET UP
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(x)

# PLOT CLUSTER
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = "red", label = "CAUTIOUS")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = "blue", label = "STANDARD")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = "green", label = "TARGET")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = "cyan", label = "CARELESS")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = "magenta", label = "CONSERVATIVE")
plt.title("TYPES OF CLIENTS")
plt.xlabel("ANNUAL INCOMES")
plt.ylabel("EXPENSE SCORE")
plt.legend()
plt.show()


# FIND # OF CLUSTER (ELBOW METHOD)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("ELBOW METHOD")
plt.xlabel("NOMBER OF CLUSTERS")
plt.ylabel("WCSS(k)")
plt.show()


# CLUSTER SET UP
kmeans = KMeans(n_clusters = 5, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# PLOT CLUSTER
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = "red", label = "CAUTIOUS")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = "blue", label = "STANDARD")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = "green", label = "TARGET")
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = "cyan", label = "CARELESS")
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = "magenta", label = "CONSERVATIVE")
plt.title("TYPES OF CLIENTS")
plt.xlabel("ANNUAL INCOMES")
plt.ylabel("EXPENSE SCORE")
plt.legend()
plt.show()
