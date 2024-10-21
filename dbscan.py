import numpy as np

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps= eps  # Maksymalna odległość do sąsiednich punktów
        self.min_samples = min_samples  # Minimalna liczba punktów w sąsiedztwie do utworzenia klastra
        self.labels_ = None  # Przechowuje etykiety klastrów

    def fit(self, X):
        n_points = len(X)
        self.labels_ = np.full(n_points, -1)  # Inicjalizacja etykiet wszystkich punktów na -1 (oznacza szum)
        cluster_id = 0

        for i in range(n_points):
            if self.labels_[i] != -1:  # Jeśli punkt już został przypisany do klastra
                continue

            # Znajdź wszystkie punkty sąsiadujące z X[i]
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                # Jeśli liczba sąsiadów jest mniejsza niż min_samples, oznacz punkt jako szum (etykieta -1)
                self.labels_[i] = -1
            else:
                # W przeciwnym razie rozpocznij nowy klaster
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """ Rozwiń nowy klaster, przypisując wszystkie punkty do klastra """
        self.labels_[point_idx] = cluster_id  # Przypisz punkt do klastra
        i = 0

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels_[neighbor_idx] == -1:
                # Jeśli sąsiad był oznaczony jako szum, przypisz go do klastra
                self.labels_[neighbor_idx] = cluster_id
            elif self.labels_[neighbor_idx] != -1:
                # Jeśli sąsiad jest już przypisany do klastra, przejdź do następnego
                i += 1
                continue

            # Znajdź nowych sąsiadów tego sąsiada
            new_neighbors = self._region_query(X, neighbor_idx)

            if len(new_neighbors) >= self.min_samples:
                # Dodaj nowych sąsiadów do listy sąsiadów
                neighbors = neighbors + new_neighbors

            i += 1

    def _region_query(self, X, point_idx):
        """ Zwróć listę sąsiadów danego punktu """
        neighbors = []
        for i, point in enumerate(X):
            if np.linalg.norm(X[point_idx] - point) < self.eps:
                neighbors.append(i)
        return neighbors

# Przykład użycia
if __name__ == "__main__":
     #Generowanie przykładowych danych (2D)
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)


    # Algorytm DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X)

    # Wizualizacja klastrów
    plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
    plt.title("DBSCAN clustering")
    plt.show()
