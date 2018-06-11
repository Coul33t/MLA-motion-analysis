from sklearn.cluster import KMeans

import numpy as np
from scipy.spatial import distance


def kmeans_algo(data, algo='k-means++', k=2, verbose=False, extrem_verbose=False):
    """
        This function is a wrapper around sklearn's KMeans function. It
        actually doesn't do anything else than applying the fit() function
        and a few other things (it shouldn't be too hard to read).
    """

    if verbose:
        print("\n\n{}\n".format(algo))

    # About fit vs fit_predict: https://stackoverflow.com/questions/25012342/scikit-learns-k-means-what-does-the-predict-method-really-do
    res = KMeans(k, algo, n_init=20, max_iter=10000, verbose=extrem_verbose).fit(list(data))

    return res


def per_cluster_inertia(data, centers, labels):
    """
        This function returns a dictionnary with its length = k (cluster number),
        containing all the cluster's inertia (opposed to sklearn's k-means,
        where it's the global inertia).
    """

    clusters_inertia = {}

    for i,center in enumerate(centers):
        cluster_data = [x for j,x in enumerate(data) if labels[j] == i]
        clusters_inertia['c' + str(i)] = sum(sum(distance.cdist([center], cluster_data, 'sqeuclidean')))

    return clusters_inertia