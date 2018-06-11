from sklearn.metrics import (average_precision_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             adjusted_mutual_info_score,
                             adjusted_rand_score)

from sklearn.metrics import (silhouette_score,
                             calinski_harabaz_score)
import numpy as np


def recall_score_computing(labels, true_labels):

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    if not isinstance(true_labels, np.ndarray):
        true_labels = np.asarray(true_labels)

    return recall_score(true_labels, labels)

def precision_score_computing(labels, true_labels):
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    if not isinstance(true_labels, np.ndarray):
        true_labels = np.asarray(true_labels)

    return precision_score(true_labels, labels)


def average_precision_score_computing(labels, true_labels):
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    if not isinstance(true_labels, np.ndarray):
        true_labels = np.asarray(true_labels)

    return average_precision_score(true_labels, labels)

def f_score_computing(labels, true_labels):
    """
        This method computes the F1-score of a 2 clusters clustering problem.
        The F1-score is defined as follow:
            F1-score = 2 * (precision * recall) / (precision + recall)
        With
            Precision = true positive / (true positive + false positive)
            Recall = true positive / (true positive + false negative)

        The values range from 0 to 1, 1 being a perfect clustering

    """

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    if not isinstance(true_labels, np.ndarray):
        true_labels = np.asarray(true_labels)


    # Since the c0 and c1 of labels may be the opposite of
    # c0 and c1 from true_labels, we check for this here.
    # If it's the case, we flip the labels in one of the array.
    similarity = len(labels) - sum(abs(labels - true_labels))

    if len(labels) - sum(abs(abs(labels - 1) - true_labels)) > similarity:
        labels = abs(labels - 1)


    return f1_score(true_labels, labels)


def adjusted_mutual_info_score_computing(labels, true_labels):
    """
        This method computes the adjusted mutual info score of a nth clusters clustering problem.

    """

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    if not isinstance(true_labels, np.ndarray):
        true_labels = np.asarray(true_labels)

    return adjusted_mutual_info_score(true_labels, labels)


def adjusted_rand_score_computing(labels, true_labels):
    """
        This method computes the adjusted rand score of a nth clusters clustering problem.
        The score ranges from 1 to -1:
            - 1        -> perfect clustering
            - 0        -> assigned labels are as good as random
            - negative -> assigned labels are worse than random (orthogonal)

    """

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    if not isinstance(true_labels, np.ndarray):
        true_labels = np.asarray(true_labels)

    return adjusted_rand_score(true_labels, labels)

def compute_all_gt_metrics(labels, true_labels):
    res = {}
    res['recall'] = recall_score_computing(labels, true_labels)
    res['precision'] = precision_score_computing(labels, true_labels)
    res['average_precision'] = average_precision_score_computing(labels, true_labels)
    res['f_score'] = f_score_computing(labels, true_labels)
    res['ami'] = adjusted_mutual_info_score_computing(labels, true_labels)
    res['ars'] = adjusted_rand_score_computing(labels, true_labels)
    return res


def silhouette_score_computing(data, labels):
    """
        This method computes the silhouette score of a nth clusters clustering problem.
        The silhouette value is a measure of how similar an object is to its own cluster
        (cohesion) compared to other clusters (separation).

        The score ranges from -1 to 1:
            - 1 -> nicely separated clusters
            - 0 -> overlapping clusters
            - negative -> indicate that a sample has been assigned to a wrong cluster
                          (a different cluster is more similar)
    """

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    return silhouette_score(data, labels)


def calinski_harabaz_score_computing(data, labels):
    """
        This method computes the Calinski and Harabaz score of a nth clusters clustering problem.
        (to maximize)

    """

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    return calinski_harabaz_score(data, labels)


def compute_all_clustering_metrics(data, labels):
    res = {}
    res['ss'] = silhouette_score_computing(data, labels)
    res['ch'] = calinski_harabaz_score_computing(data, labels)
    return res