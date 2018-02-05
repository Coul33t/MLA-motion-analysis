from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import numpy as np
from scipy.spatial import distance
import pdb

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
        This function returns a list with its length = k (cluster number),
        containing all the cluster's inertia (opposed to sklearn's k-means,
        where it's the global inertia). Of course, the data must be sorted 
        in the same order as the labels' list, i.e.
        data[0] -> labels[0], data[1] -> labels[1], etc.
    """

    clusters_inertia = np.zeros( (len(centers) ))

    for i,center in enumerate(centers):
        cluster_data = [x for j,x in enumerate(data) if labels[j] == i]
        clusters_inertia[i] += sum(sum(distance.cdist([center], cluster_data, 'sqeuclidean')))


    return clusters_inertia


def f_score_computing(labels, true_labels):
    """
        This methode returns the F1-score of a 2 clusters clustering problem.
        The F1-score is defined as follow:
            F1-score = 2 * (precision * recall) / (precision + recall)
        With    
            Precision = true positive / (true positive + false positive)
            Recall = true positive / (true positive + false negative)

        For the moment, since I'm unable to think of a nice way to permute
        the clusters' labels for more than 2 clusters, it only works for
        2 clusters.

    """

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    if not isinstance(true_labels, np.ndarray):
        true_labels = np.asarray(true_labels)

    # 2 cases here: either true_labels is the label for each data,
    # in which case len(labels) == len(true_labels), OR it is the
    # array (natural indexes) telling where the success are; thus
    # the need of a transformation
    if len(labels) != len(true_labels):
        tmp_true_labels = np.zeros( (len(labels)) )
        # Switching from natural indexes to array indexes
        tmp_true_labels[true_labels- 1] = 1
        true_labels = tmp_true_labels.astype(int)


    # Since the c0 and c1 of labels may be the opposite of
    # c0 and c1 from true_labels, we check for this here.
    # If it's the case, we flip the labels in one of the array.
    similarity = len(labels) - sum(abs(labels - true_labels))

    if len(labels) - sum(abs(abs(labels - 1) - true_labels)) > similarity:
        labels = abs(labels - 1)


    return f1_score(true_labels, labels)