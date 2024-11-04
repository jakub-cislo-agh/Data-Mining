from MySCAN import *
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# przykładowy zbiór testowy z sklearn.dataset
X, _ = make_blobs(n_samples=100, centers=3, random_state=37)

# tutaj sobie klasteryzujemy
dbscan = MySCAN(eps=1, min_pts=5) # tutaj podajemy parametry eps i min_pts
dbscan.clusterize(X)

#dbscan z biblioteki na tym samym zbiorze
dbscan_lib = DBSCAN(eps = 1, min_samples=5).fit(X)

#tworzymy dwa wykresy
fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
ax[0].set_title("Klasteryzacja DBSCAN")

ax[1].scatter(X[:, 0], X[:, 1], c=dbscan_lib.labels_)
ax[1].set_title("Klasteryzacja biblioteczna DBSCAN")

plt.show()