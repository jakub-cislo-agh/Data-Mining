import unittest
import numpy as np
from MySCAN import MySCAN

class TestDBSCAN(unittest.TestCase):
    def setUp(self):
        # inicjalizacja przykładowych danych do testów
        self.X = np.array([
            [1, 2], [2, 2], [2, 3],  # Klastry blisko siebie
            [8, 7], [8, 8], [25, 80]  # szum
        ])
        self.eps = 1.5
        self.min_samples = 2
        self.dbscan = MySCAN(eps=self.eps, min_pts=self.min_samples)

    def test_find_neighbors(self):
        # testowanie `_find_neighbors` na obecność sąsiadów
        neighbors = self.dbscan._find_neighbors(self.X, 0)
        expected_neighbors = [0, 1, 2]
        self.assertCountEqual(neighbors, expected_neighbors)

    def test_expand_cluster(self):
        # Wywołujw `clusterize`, aby zainicjalizować `labels_`
        self.dbscan.clusterize(self.X)
        
        # testowanie `_expand_cluster`, czy dobrze przypisuje etykiety do klastrów
        neighbors = [0, 1, 2]
        self.dbscan._expand_cluster(self.X, 0, neighbors, cluster_id=0)
        
        # sprawdzenie, czy wszystkie punkty należą do klastra o identyfikatorze 0
        for idx in neighbors:
            self.assertEqual(self.dbscan.labels_[idx], 0)

    def test_clusterize(self):
        self.dbscan.clusterize(self.X)
        labels = self.dbscan.labels_
        
        # sprawdzenie, czy punkty [1, 2], [2, 2], [2, 3] należą do tego samego klastra
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])

        # Sprawdzenie, czy punkty [8, 7] i [8, 8] należą do tego samego klastra
        self.assertEqual(labels[3], labels[4])

        # Sprawdzenie, czy punkt [25, 80] został oznaczony jako szum (etykieta -1)
        self.assertEqual(labels[5], -1)

    def test_noise_points(self):
        # Testowanie, czy punkty nie spełniające kryteriów są oznaczone jako szum
        X = np.array([[0, 0], [10, 10]])  # outliers
        dbscan = MySCAN(eps=1, min_pts=2)
        dbscan.clusterize(X)
        
        # czy oba punkty zostały oznaczone jako szum
        np.testing.assert_array_equal(dbscan.labels_, [-1, -1])

if __name__ == "__main__":
    unittest.main() # Output: "Ran 4 tests in 0.009s. OK"
