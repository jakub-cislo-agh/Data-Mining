import numpy as np

class MySCAN:
    def __init__(self, eps, min_pts):
        self.eps = eps  # maksymalny promień sąsiedztwa
        self.min_pts = min_pts  # minimalna liczba obiektów wchodząca w skład klastra
        self.labels_ = None  # Przechowuje etykiety klastrów

    def clusterize(self, X):
        n_points = len(X)
        self.labels_ = np.full(n_points, -1)  # inicjalizacja etykiet wszystkich punktów na -1 (szum)
        cluster_id = 0

        for i in range(n_points):
            if self.labels_[i] != -1:  # pomijamy, jeżeli punkt już został przypisany do klastra
                continue

            neighbors = self._find_neighbors(X, i) # szukamy sąsiadów X[i]

            if len(neighbors) < self.min_pts: # jeśli liczba sąsiadów jest mniejsza niż min_pts, oznacz punkt jako szum
                self.labels_[i] = -1
            else: # w przeciwnym razie rozpocznij nowy klaster
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id): #przypisuje wszystkie punkty do klastra
        self.labels_[point_idx] = cluster_id  # Przypisz punkt do klastra
        i = 0

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels_[neighbor_idx] == -1: # jeżeli punkt nie był jeszcze przypisany, przypisujemy do klastra
                self.labels_[neighbor_idx] = cluster_id
            else: # w przeciwnym przypadku, przejdź do następnego
                i += 1
                continue

            new_neighbors = self._find_neighbors(X, neighbor_idx) # szukamy nowych sąsiadów tego sąsiada

            if len(new_neighbors) >= self.min_pts: # dodajemy nowych sąsiadów do listy sąsiadów
                neighbors = neighbors + new_neighbors

            i += 1

    def _find_neighbors(self, X, point_idx): # zwraca listę sąsiadów danego punktu
        neighbors = []
        for i, point in enumerate(X):
            if np.linalg.norm(X[point_idx] - point) < self.eps:
                neighbors.append(i)
        return neighbors
