import numpy as np
from MySCAN import MySCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets

# wczytanie danych przy pomocy load_digits() z sklearn.datasets, czyli zredukowany MNIST
mnist = datasets.load_digits()
X_data, Y_data = mnist['data'], mnist['target']

# redukcja wymiaru dla lepszej efektywności DBSCAN (opcjonalne)
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_data)


# Parametry DBSCAN - warto je dostosować
min_samples = 12
eps = 23
dbscan = MySCAN(eps=eps, min_pts=min_samples)

# Analiza odległości najbliższego sąsiada
# neighbors = NearestNeighbors(n_neighbors=min_samples)
# neighbors_fit = neighbors.fit(X_reduced)
# distances, indices = neighbors_fit.kneighbors(X_reduced)

# distances = np.sort(distances[:, min_samples - 1], axis=0)

# Wykres
# plt.figure(figsize=(10, 6))
# plt.plot(distances)
# plt.title('Distances to the {}th nearest neighbor'.format(min_samples))
# plt.xlabel('Points sorted by distance')
# plt.ylabel('Distance')
# plt.grid()
# plt.show()

# Uruchomienie klasteryzacji
dbscan.clusterize(X_reduced)

# wyświetl wyniki klasteryzacji
unique_labels = np.unique(dbscan.labels_)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = list(dbscan.labels_).count(-1)

print(f"Liczba wykrytych klastrów: {n_clusters}")
print(f"Liczba punktów oznaczonych jako szum: {n_noise}")

# Ewaluacja klasteryzacji
ari = adjusted_rand_score(Y_data, dbscan.labels_)
print("ARI:", ari) #ARI = 0.4718

# Wizualizacja wybranych klastrów po redukcji do 2 wymiarów
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_reduced)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=dbscan.labels_, cmap='Paired', s=5)
plt.title("DBSCAN clustering on MNIST subset")
plt.show()
