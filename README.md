MySCAN.py contains a custom built MySCAN(eps, min_pts) class \n
eps – the maximum distance between two samples for one to be considered as in the neighborhood of the other
min_pts – The number of samples in a neighborhood for a point to be considered as a core point

MySCAN can be used to perform DBSCAN clustering on a given dataset
To do that use MySCAN.clusterize(X) where X is the given dataset

Example of using MySCAN with visualisation of the clusters is in main.py

compare-dbscan.py and compare-k-means.py show visual comparison of our MySCAN to scikit-learn clusterization algorithms

