from MySCAN import *
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# przykładowy zbiór testowy z sklearn.dataset
X, _ = make_blobs(n_samples=100, centers=3, random_state=37)

# tutaj sobie klasteryzujemy
dbscan = MySCAN(eps=1, min_pts=5) # tutaj podajemy parametry eps i min_pts
dbscan.clusterize(X)

# wizualizacja klastrów
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
plt.title("Klasteryzacja DBSCAN")
plt.show()