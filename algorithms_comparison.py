from MySCAN import *
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import adjusted_rand_score

# wczytanie danych przy pomocy load_digits() z sklearn.datasets, czyli zredukowany MNIST
mnist = datasets.load_digits()
X_data, Y_data = mnist['data'], mnist['target']

# redukcja wymiaru dla lepszej efektywności DBSCAN (opcjonalne)
pca = PCA(n_components=50)
X = pca.fit_transform(X_data)

# klasteryzacja naszym algorytmem
dbscan = MySCAN(eps=23, min_pts=12) # tutaj podajemy parametry eps i min_pts
dbscan.clusterize(X)

#dbscan z biblioteki na tym samym zbiorze
dbscan_lib = DBSCAN(eps = 23, min_samples=12).fit(X)

#k-means z biblioteki na tym samym zbiorze
kmeans_lib = KMeans(n_clusters = 10).fit(X)

#tworzymy wykresy, by wizualnie porównać algorytmy klasteryzacji
fig, ax = plt.subplots(nrows=1, ncols=3)

rating = adjusted_rand_score(dbscan.labels_, dbscan_lib.labels_)
print(rating)

ax[0].scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
ax[0].set_title("Klasteryzacja DBSCAN")

ax[1].scatter(X[:, 0], X[:, 1], c=dbscan_lib.labels_)
ax[1].set_title("Klasteryzacja biblioteczna DBSCAN")

ax[2].scatter(X[:, 0], X[:, 1], c=kmeans_lib.labels_)
ax[2].set_title("Klasteryzacja biblioteczna K-means")

plt.show()

# Ewaluacja klasteryzacji
ari_db = adjusted_rand_score(Y_data, dbscan.labels_)
print("MyScan ARI:", ari_db) #ARI = 0.4718

ari_lib_db = adjusted_rand_score(Y_data, dbscan_lib.labels_)
print("DBSCAN ARI:", ari_lib_db) #ARI = 0.4718

ari_lib_kmeans = adjusted_rand_score(Y_data, kmeans_lib.labels_)
print("K-means ARI:", ari_lib_kmeans) #ARI = 0.6349

#Klasteryzacja K-means radzi sobie z tym zadaniem lepiej