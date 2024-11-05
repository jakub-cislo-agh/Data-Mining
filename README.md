_MySCAN.py_ contains a custom built _MySCAN(eps, min_pts)_ class  
_eps_ – the maximum distance between two samples for one to be considered as in the neighborhood of the other  
_min_pts_ – The number of samples in a neighborhood for a point to be considered as a core point

_MySCAN_ can be used to perform DBSCAN clustering on a given dataset  
To do that, use _MySCAN.clusterize(X)_ where _X_ is the given dataset

You can find an example of using _MySCAN_ with visualisation of the clusters in _main.py_

_compare-dbscan.py_ and _compare-k-means.py_ show visual comparison of our _MySCAN_ to scikit-learn clusterization algorithms
