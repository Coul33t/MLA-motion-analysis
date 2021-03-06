# Native packages
import os
import re
# TODO: remove when Python 3.7 is out (dict will keep the insertion value)
from collections import OrderedDict

import json

import pdb

# External lib packages
import numpy as np

from sklearn.cluster import estimate_bandwidth
from sklearn.preprocessing import RobustScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA

# Personnal packages
from tools import (flatten_list,
                   motion_dict_to_list,
                   natural_keys,
                   string_length_shortening,
                   string_redundancy_remover)

# Data import functions
from data_import import (return_files,
                         json_import,
                         json_specific_import)

# Clustering algorithms and metrics computing
from algos.kmeans_algo import (kmeans_algo,
                               kmeans_algo_2,
                               per_cluster_inertia)

from algos.dbscan_algo import (dbscan_algo)

from algos.agglomerative_algo import (agglomerative_algo)

from algos.mean_shift_algo import (mean_shift_algo)

from algos.gmm_algo import (gmm_algo)

from algos.metrics import (f_score_computing,
                           adjusted_mutual_info_score_computing,
                           adjusted_rand_score_computing,
                           silhouette_score_computing,
                           calinski_harabaz_score_computing,
                           compute_all_clustering_metrics,
                           compute_all_gt_metrics)

# Data visualization functions
from data_visualization import (plot_data_k_means,
                                plot_data_sub_k_means,
                                simple_plot_2d,
                                simple_plot_2d_2_curves,
                                plot_all_defaults_at_once)

from data_selection import data_gathering

# Results class
from results_analysing import Results

# Ground truth for data
import data_labels as dl

from feedback_tools import *
# Constants (well more like " huge lists " but whatever)
import constants as cst

def joint_selection(data, joints_to_append):
    """
        This function returns a list of desired joints data,
        from a list of Motion objects.
    """

    # We gonna switch from dict to OrderedDict, to ensure that each
    # features vector has the same joints order
    selected_data = []

    datatypes = data[0].get_datatypes_names()

    # If joints_to_append isn't specified, we take all the joints. Just to be
    # sure that the order won't change between 2 motion dicts, we take the keys
    # of one motion and use it as the reference for insertion order
    if not joints_to_append:
        joints_to_append = data[0].get_joint_list()

    for motion in data:
        # We keep the name
        selected_joints_motion = [motion.name, OrderedDict()]

        # For each datatype
        for datatype in datatypes:

            # We use an OrderedDict. Since we iterate over the joints_to_append
            # list, the order is ensured to be the same for each motion
            joints_selected = OrderedDict()

            # For each joint to append
            if isinstance(joints_to_append, list):
                for joint in joints_to_append:
                    if joint not in motion.get_joint_list():
                        joint = motion.laterality + joint


                    joints_selected[joint] = motion.get_datatype(datatype).get_joint_values(joint)


            else:
                if joints_to_append not in motion.get_joint_list():
                    joints_to_append = motion.laterality + joints_to_append

                joints_selected[joints_to_append] = motion.get_datatype(datatype).get_joint_values(joints_to_append)

            selected_joints_motion[1][datatype] = joints_selected

        selected_data.append(selected_joints_motion)

    return selected_data


def data_selection(data, data_to_keep):
    """
        This function returns an array of desired features.
    """

    features = []

    # For each motion
    # motion[0] is the filename
    # motion[1] are the values, ordered by joints and datatypes
    # (datatype->joints->values)
    for motion in data:
        motion_feat = OrderedDict()

        # For each data type we want to gather
        for datatype in data_to_keep:

            # Don't ask how it works (= nb_frames)
            nb_values = len(list(motion[1][datatype].values())[0])

            for i in range(nb_values):
                # For each joint
                for joint in motion[1][datatype]:
                    # If it doesn't exists, create an empty list then append
                    # Else, append

                    motion_feat.setdefault(joint, []).append(motion[1][datatype][joint][i])

        features.append(flatten_list(list(motion_feat.values())))

    # We return the list as a numpy array, as it is more
    # convenient to use
    return np.asarray(features)


def check_algo_parameters(algo, parameters):
    clean_parameters = {}
    if algo == 'k-means':
        clean_parameters['n_clusters']  = 2 if 'n_clusters' not in parameters else parameters['n_clusters']
        clean_parameters['init']        = 'k-means++' if 'init' not in parameters else parameters['init']
        clean_parameters['n_init']      = 10 if 'n_init' not in parameters else parameters['n_init']
        clean_parameters['tol']         = 1e-4 if 'tol' not in parameters else parameters['tol']
        clean_parameters['verbose']     = 0 if 'verbose' not in parameters else parameters['verbose']

    elif algo == 'dbscan':
        clean_parameters['eps']             = 0.5 if 'eps' not in parameters else parameters['eps']
        clean_parameters['min_samples']     = 5 if 'min_samples' not in parameters else parameters['min_samples']
        clean_parameters['metric']          = 'euclidean' if 'metric' not in parameters else parameters['metric']
        clean_parameters['algorithm']       = 'auto' if 'algorithm' not in parameters else parameters['algorithm']

    elif algo == 'agglomerative':
        clean_parameters['n_clusters']  = 2 if 'n_clusters' not in parameters else parameters['n_clusters']
        clean_parameters['affinity']    = 'euclidean' if 'affinity' not in parameters else parameters['affinity']
        clean_parameters['linkage']     = 'ward' if 'linkage' not in parameters else parameters['linkage']

    elif algo == 'mean-shift':
        clean_parameters['quantile']    = 0.3 if 'quantile' not in parameters else parameters['quantile']
        clean_parameters['cluster_all'] = False if 'cluster_all' not in parameters else parameters['cluster_all']

    elif algo == 'gmm':
        clean_parameters['n_components']    = 1 if 'n_components' not in parameters else parameters['n_components']
        clean_parameters['covariance_type'] = 'full' if 'covariance_type' not in parameters else parameters['covariance_type']
        clean_parameters['tol']             = 1e-3 if 'tol' not in parameters else parameters['tol']
        clean_parameters['max_iter']        = 100 if 'max_iter' not in parameters else parameters['max_iter']
        clean_parameters['n_init']          = 1 if 'n_init' not in parameters else parameters['n_init']

    return clean_parameters


def distance_matrix_computing(data):
    """
        Compute the euclidean distance between each sample of the data.
    """

    distance_matrix = np.zeros((len(data), len(data)))

    for i, _ in enumerate(data):
        for j, _ in enumerate(data):
            distance_matrix[i][j] = abs(sum(data[i] - data[j]))

    return distance_matrix


def compute_mean_std(d_m, success_indexes, natural_indexes=False):
    """
        Compute the mean and std of distance between features.
    """
    success_indexes = np.asarray(success_indexes)

    # If the indexes are given in natural order (1 is the first value index),
    # we shift them to fit the array idx (0 is the first value index)
    if natural_indexes:
        success_indexes = success_indexes - 1

    # extract the lines
    success_distance_matrix = []
    for idx in success_indexes:
        success_distance_matrix.append(d_m[idx])
    success_distance_matrix = np.asarray(success_distance_matrix)

    failure_distance_matrix = []
    fail_idx = [x for x in range(0, 100) if x not in success_indexes]
    for idx in fail_idx:
        failure_distance_matrix.append(d_m[idx])
    failure_distance_matrix = np.asarray(failure_distance_matrix)

    # Mask corresponding to the success
    success_mask = np.ones(len(d_m), dtype=bool)
    success_mask[success_indexes] = False
    # Mask corresponding to the failures
    failure_mask = np.ones(len(d_m), dtype=bool)
    failure_mask[fail_idx] = False

    cutted_success_distance_matrix = np.zeros((len(success_distance_matrix), len(success_distance_matrix)))
    cutted_failure_distance_matrix = np.zeros((len(failure_distance_matrix), len(failure_distance_matrix)))

    for i, _ in enumerate(success_distance_matrix):
        cutted_success_distance_matrix[i] = success_distance_matrix[i][failure_mask]

    for i, _ in enumerate(failure_distance_matrix):
        cutted_failure_distance_matrix[i] = failure_distance_matrix[i][success_mask]

    return [np.mean(cutted_success_distance_matrix),
            np.std(cutted_success_distance_matrix),
            np.mean(cutted_failure_distance_matrix),
            np.std(cutted_failure_distance_matrix)]

def k_mean_mixed(path, original_data, name, validate_data=False,
                          joint_to_use=None, data_to_select=None,
                          true_labels=None, verbose=False, to_file=True,
                          to_json=True, display_graph=False,
                          save_graph=False, data_to_graph=None):
    """
        This function run a k-means algorithm with varying k values,
        on each joint.
    """

    if (display_graph or save_graph) and not data_to_graph:
        print('ERROR: no data specified for graph output.')
        return

    if validate_data:
        for motion in original_data:
            motion.validate_motion()

    # If there's no specific datatype defined, we take all the data available
    if not data_to_select:
        data_to_select = set([name for motion in original_data for name in motion.datatypes])

    print('\nData used: {}'.format(data_to_select))

    # If there's no joint to select, then we take all of them
    if joint_to_use is None:
        joint_to_use = original_data[0].get_joint_list()

    print('Joints used: {}\n'.format(joint_to_use))

    # This OrderedDict will contain each joint as a key, and for each
    # joint, a list of list, [nb_cluster, inertia]
    res_k = OrderedDict()

    results = Results()

    for k in range(2, 11):
        print('Running for k = {}'.format(k))

        # For each joint combination
        for joint in joint_to_use:
            joint_name = joint

            if isinstance(joint, list):
                joint_name = ','.join(joint)

            # We keep the joints' data we're interested in
            # (from the motion class)
            selected_data = joint_selection(original_data, joint)

            # We select the data we want and we put them in the right shape
            # for the algorithm
            # [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
            features = data_selection(selected_data, data_to_select)

            # Actual k-means
            res = kmeans_algo(features, k=k)

            metrics = {}
            # Computing the f1-score, adjusted mutual information score
            # and adjusted rand score, if the number of clusters correspond
            # to the number of clusters in the ground truth
            # If we're working only with the success, we have no ground truth
            # so we can't ompute these scores
            if true_labels and k == len(np.unique(true_labels)):
                metrics.update(compute_all_gt_metrics(res.labels_, true_labels))

            metrics.update(compute_all_clustering_metrics(features, res.labels_))

            if verbose:
                print('Joint: {}'.format(joint))
                print("Global inertia: {}".format(res.inertia_))
                clusters_composition(res.labels_, true_labels, len(original_data), verbose=True)

            # Computing the inertia for each cluster
            clusters_inertia = per_cluster_inertia(features, res.cluster_centers_, res.labels_)

            # If we have the true labels (currently only works for c = 2 since
            # I don't know how to test every permutation possible)
            if true_labels and len(set(true_labels)) == 2:
                c_comp = clusters_composition(res.labels_, true_labels, len(original_data), verbose=False)
            else:
                c_comp = clusters_composition_name(res.labels_, len(original_data),
                                                   original_names=np.asarray([o.name for o in original_data]),
                                                   verbose=False)

            # Appending the value for the current k to the results OrderedDict
            # (used for graph)
            if data_to_graph and data_to_graph in metrics.keys():
                res_k.setdefault(joint_name, []).append([k, metrics[data_to_graph]])

            # Compute the distance between each centroids for each dimension
            centroids = np.asarray(res.cluster_centers_)
            centroids_diff = {}
            for i in range(len(centroids) - 1):
                for j in range(i+1, len(centroids)):
                    centroids_diff[f'c{i}_c{j}'] = abs(centroids[i] - centroids[j]).tolist()

            # Add the current algorithm output to the results
            results.add_values(k=k, data_used=data_to_select, joints_used=joint, centroids=res.cluster_centers_,
                               centroids_diff=centroids_diff, global_inertia=res.inertia_,
                               per_cluster_inertia=clusters_inertia, metrics=metrics, motion_repartition=c_comp)


    path_to_export = ''
    # If the length of the folder is too long, we shorten it
    folder_name = string_length_shortening(name, max_size=50)
    path_to_export = r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + folder_name +'/'


    data_to_export = 'all'
    # data_to_export = results.get_res(ars=[0.1, 'supeq'])

    # Exporting the data
    results.export_data(path=path_to_export,
                        data_to_export=data_to_export,
                        text_export=to_file,
                        json_export=to_json)

    # Plotting or saving the desired values
    if display_graph or save_graph:
        plot_save_name = ''

        if save_graph:
            plot_save_name = "_".join(data_to_select)

            if true_labels:
                plot_save_name += '_c' + str(len(set(true_labels)))

        plot_save_name = string_redundancy_remover(plot_save_name)

        plot_data_k_means(res_k, display=display_graph, save=save_graph,
                          name=plot_save_name, path=path_to_export,
                          graph_title=plot_save_name.replace('_', ' '),
                          x_label='k value', y_label=data_to_graph)

    return results

def test_full_batch_k_var(path, original_data, name, validate_data=False,
                          joint_to_use=None, data_to_select=None,
                          true_labels=None, verbose=False, to_file=True,
                          to_json=True, display_graph=False,
                          save_graph=False, data_to_graph=None,
                          only_success=False):
    """
        This function run a k-means algorithm with varying k values,
        on each joint.
    """
    # To extract the succesful motion only, we need the ground truth with c = 2
    # (c0 = failure, c1 = success)
    if only_success and (not true_labels or len(set(true_labels)) != 2):
        print('ERROR: must have true labels with c=2 to extract the succesful motions only.')
        return

    if (display_graph or save_graph) and not data_to_graph:
        print('ERROR: no data specified for graph output.')
        return

    if validate_data:
        for motion in original_data:
            motion.validate_motion()

    # If there's no specific datatype defined, we take all the data available
    if not data_to_select:
        data_to_select = set([name for motion in original_data for name in motion.datatypes])

    print('\nData used: {}'.format(data_to_select))

    # If there's no joint to select, then we take all of them
    if joint_to_use is None:
        joint_to_use = original_data[0].get_joint_list()

    print('Joints used: {}\n'.format(joint_to_use))

    # This OrderedDict will contain each joint as a key, and for each
    # joint, a list of list, [nb_cluster, inertia]
    res_k = OrderedDict()

    # simple_plot_2d_2_curves(original_data[0].get_datatype('Norm').get_joint('LeftHand'),
    #                         original_data[0].get_datatype('SavgoledNorm').get_joint('LeftHand'))

    # Initialising
    for joint in joint_to_use:
        # If it's a combination of joints
        if isinstance(joint, list):
            res_k[','.join(joint)] = []
        else:
            res_k[joint] = []

    results = Results()

    # For each k value (2 - 10)
    for k in range(2, 11):
        print('Running for k = {}'.format(k))

        # For each joint combination
        for joint in joint_to_use:
            joint_name = joint

            if isinstance(joint, list):
                joint_name = ','.join(joint)

            # We keep the joints' data we're interested in
            # (from the motion class)
            selected_data = joint_selection(original_data, joint)

            # We select the data we want and we put them in the right shape
            # for the algorithm
            # [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
            features = data_selection(selected_data, data_to_select)

            # If we only work with the succesful motions,
            # we only keep these ones (c == 1 for success)
            if only_success:
                features = features[(np.asarray(true_labels) == 1)]

            # Compute the euclidean distance between each sample
            # (Unused in this version)
            # d_m = distance_matrix_computing(features)
            #
            # c_res = compute_mean_std(d_m, GLOUP_SUCCESS, natural_indexes=True)

            # Actual k-means
            res = kmeans_algo(features, k=k)

            metrics = {}
            # Computing the f1-score, adjusted mutual information score
            # and adjusted rand score, if the number of clusters correspond
            # to the number of clusters in the ground truth
            # If we're working only with the success, we have no ground truth
            # so we can't compute these scores
            if not only_success and true_labels and k == len(np.unique(true_labels)):
                metrics.update(compute_all_gt_metrics(res.labels_, true_labels))

            metrics.update(compute_all_clustering_metrics(features, res.labels_))

            if verbose:
                print('Joint: {}'.format(joint))
                print("Global inertia: {}".format(res.inertia_))
                clusters_composition(res.labels_, true_labels, len(original_data), verbose=True)

            # Computing the inertia for each cluster
            clusters_inertia = per_cluster_inertia(features, res.cluster_centers_, res.labels_)

            # If we have the true labels (currently only works for c = 2 since
            # I don't know how to test every permutation possible)
            if true_labels and len(set(true_labels)) == 2:
                c_comp = clusters_composition(res.labels_, true_labels, len(original_data), verbose=False)
            else:
                c_comp = clusters_composition_name(res.labels_, len(original_data),
                                                   original_names=np.asarray([o.name for o in original_data]),
                                                   verbose=False)

            # Appending the value for the current k to the results OrderedDict
            # (used for graph)
            if data_to_graph and data_to_graph in metrics.keys():
                res_k[joint_name].append([k, metrics[data_to_graph]])

            # Compute the distance between each centroids for each dimension
            centroids = np.asarray(res.cluster_centers_)
            centroids_diff = {}
            for i in range(len(centroids) - 1):
                for j in range(i+1, len(centroids)):
                    centroids_diff[f'c{i}_c{j}'] = abs(centroids[i] - centroids[j]).tolist()

            # Add the current algorithm output to the results
            results.add_values(k=k, data_used=data_to_select, joints_used=joint, centroids=res.cluster_centers_,
                               centroids_diff=centroids_diff, global_inertia=res.inertia_,
                               per_cluster_inertia=clusters_inertia, metrics=metrics, motion_repartition=c_comp)

    # for key, val in res_k.items():
    #     max_val = -9999
    #     k = 0

    #     for elem in val:
    #         if elem[1] > max_val:
    #             max_val = elem[1]
    #             k = elem[0]

    #     print(f"{key}: {max_val}({k})")

    if verbose:
        print("Low inertia joints (< 50):{}\n".format(low_inertia))
        print("High inertia joints (>= 50):{}\n".format(high_inertia))
        print("Intersection:{}\n".format(low_inertia & high_inertia))

    path_to_export = ''
    # If the length of the folder is too long, we shorten it
    folder_name = string_length_shortening(name, max_size=50)
    path_to_export = r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + folder_name +'/'


    data_to_export = 'all'
    # data_to_export = results.get_res(ars=[0.1, 'supeq'])

    # Exporting the data
    results.export_data(path=path_to_export,
                        data_to_export=data_to_export,
                        text_export=to_file,
                        json_export=to_json)

    # Plotting or saving the desired values
    if display_graph or save_graph:
        plot_save_name = ''

        if save_graph:
            plot_save_name = "_".join(data_to_select)

            if only_success:
                plot_save_name += '_only_success'
            if true_labels:
                plot_save_name += '_c' + str(len(set(true_labels)))

        plot_save_name = string_redundancy_remover(plot_save_name)

        plot_data_k_means(res_k, display=display_graph, save=save_graph,
                          name=plot_save_name, path=path_to_export,
                          graph_title=plot_save_name.replace('_', ' '),
                          x_label='k value', y_label=data_to_graph)

def run_clustering(original_data, validate_data=False,
                   datatype_joints=None, algorithm='k-means',
                   parameters={}, scale_features=False, normalise_features=False,
                   true_labels=None, verbose=False, to_file=True, to_json=True, return_data=False):

    """
        This function run a k-means algorithm with varying k values,
        on each joint.
    """

    if algorithm not in cst.implemented_algo:
        print(f'ERROR: {algorithm} not implemented (yet).')
        return

    if validate_data:
        for motion in original_data:
            motion.validate_motion()

    # If there's no specific datatype defined, we take all the data available
    # if not data_to_select:
    #     data_to_select = set([name for motion in original_data for name in motion.datatypes])

    if verbose:
        print('\nData used: {}'.format(data_to_select))

    # If there's no joint to select, then we take all of them
    # if joint_to_use is None:
    #     joint_to_use = original_data[0].get_joint_list()

    if verbose:
        print('Joints used: {}\n'.format(joint_to_use))

    results = Results()

    # We keep the joints' data we're interested in
    # (from the motion class)
    # selected_data = joint_selection(original_data, joint_to_use)

    # We select the data we want and we put them in the right shape
    # for the algorithm
    # [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
    # features = data_selection(selected_data, data_to_select)

    features = data_gathering(original_data, datatype_joints)

    # Compute the euclidean distance between each sample
    # (Unused in this version)
    # d_m = distance_matrix_computing(features)
    #
    # c_res = compute_mean_std(d_m, GLOUP_SUCCESS, natural_indexes=True)

    if (scale_features):
        scaler = RobustScaler()
        features = scaler.fit_transform(features)

    if (normalise_features):
        min_max_scaler = MinMaxScaler()
        features = min_max_scaler.fit_transform(features)


    params = check_algo_parameters(algorithm, parameters)

    if algorithm == 'k-means':
        # Actual k-means
        res = kmeans_algo_2(features, params)

    elif algorithm == 'dbscan':
        res = dbscan_algo(features, params)

    elif algorithm == 'agglomerative':
        res = agglomerative_algo(features, params)

    elif algorithm == 'mean-shift':
        if 'quantile' in params:
            estimated_bw = estimate_bandwidth(features, params['quantile'])

            if estimated_bw > 0:
                params['bandwidth'] =  estimated_bw

            params.pop('quantile')

        res = mean_shift_algo(features, params)

    elif algorithm == 'gmm':
        res = gmm_algo(features, params)

    metrics = {}
    # Computing the f1-score, adjusted mutual information score
    # and adjusted rand score, if the number of clusters correspond
    # to the number of clusters in the ground truth
    # If we're working only with the success, we have no ground truth
    # so we can't compute these scores
    if true_labels and params['n_clusters'] == len(np.unique(true_labels)):
        metrics.update(compute_all_gt_metrics(res.labels_, true_labels))

    metrics.update(compute_all_clustering_metrics(features, res.labels_))

    # Features are returned to be plotted
    if return_data:
        return[res, metrics, features]

    return [res, metrics]

def clusters_composition(labels, true_labels, sample_nb, verbose=False):
    """
        For each k in labels, this function returns the clusters' composition.
        TODO: redo
    """

    # Get a list of clusters number
    cluster_nb = set(labels)

    success = np.asarray(dl.DBRUN_SUCCESS)
    # Switching from natural idx to array idx
    success = success - 1

    failure = [x for x in range(0, sample_nb) if x not in success]

    c_composition = {}

    for k in cluster_nb:
        # use " labels.index(k) " ??? -> not with numpy arrays
        sample_idx = [i for i, x in enumerate(labels) if x == k]

        c_composition['success in c' + str(k)] = len(set(success) & set(sample_idx))
        c_composition['failure in c' + str(k)] = len(set(failure) & set(sample_idx))
        # YAPAT
        # set(success) & set(sample_idx) -> gives the intersection of the 2 sets
        # (values that are in the 2 lists)
        if verbose:
            print("Success in c{}: {} / Failure in c{}: {}".format(k, len(set(success) & set(sample_idx)),
                                                                   k, len(set(failure) & set(sample_idx))))
    return c_composition

def clusters_composition_name(labels, sample_nb, original_names, verbose=False):
    """
        For each k in labels, this function returns the clusters' composition (filenames).
    """

    c_composition = {}

    # Get a list of clusters number
    cluster_nb = set(labels)

    # Used to strip off the " Char00 " from the filename
    regex = re.compile(r'^\w*_\d+')

    stripped_names = np.asarray([regex.search(s).group() for s in original_names])

    # For each cluster
    for c in cluster_nb:
        c_composition['c' + str(c)] = stripped_names[np.where(labels == c)].tolist()

        if verbose:
            print("\nMotions in c{}: {}".format(c, stripped_names[np.where(labels == c)]))

    return c_composition

def plot_speed(path, file, joint_to_use):
    original_data = json_import(path, file)

    data_to_select = ['SpeedNorm']

    pdb.set_trace()
    # We keep the data we're interested in (from the .json file)
    selected_data = joint_selection(original_data, joint_to_use)

    # We put them in the right shape for the algorithm [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
    features = data_selection(selected_data, data_to_select)

    simple_plot_2d(features)

def k_means_second_pass(file_path, result_path, file_name, person_name):
    """
        Make a second pass on sub-clusters.
    """

    # What a nice idea to also export results as json files...
    # Why does it produce a list instead of just the dic tho
    res_total = json.load(open(result_path + file_name))

    # For every good combination (ss > 0.5)
    for one_res in res_total:
        print(f'Processing for {one_res["k"]} clusters')
        data_sub_separation = []

        # We get the different motion repartition
        for cluster in one_res['motion_repartition'].keys():
            data_sub_separation.append([name+'Char00' for name in one_res['motion_repartition'][cluster]])

        # For each same cluster motion, we do another pass of the algorithm
        # idx for export name
        for idx, subcomp in enumerate(data_sub_separation):
            sub_data = json_specific_import(file_path, subcomp)

            # Preparing some variables for the rest of the algorithm
            # It's acutally more readable to have explicit variable names
            # instead of dictionnary keys
            joint = one_res['joints_used']
            if ',' in one_res['joints_used']:
                joint = one_res['joints_used'].replace(' ', '').split(',')

            data_to_select = one_res['data_used']
            if ',' in one_res['data_used']:
                data_to_select = one_res['data_used'].replace(' ', '').split(',')
            else:
                data_to_select = [data_to_select]

            data_to_graph = 'ss'

            # This list will contains the result for each k
            res_k = []

            results = Results()

            max_k_value = 10

            if len(sub_data) <= max_k_value:
                max_k_value = len(sub_data) - 1

            # No need to compute a k-means for a sample number = 2
            if max_k_value <= 2:
                continue

            print(f'Running on c{idx} for k = 2 to {max_k_value}')
            for k in range(2, max_k_value + 1):

                 # We keep the joints' data we're interested in (from the motion class)
                selected_data = joint_selection(sub_data, joint)

                # We select the data we want and we put them in the right shape
                # for the algorithm [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
                features = data_selection(selected_data, data_to_select)

                # Actual k-means
                res = kmeans_algo(features, k=k)

                # Computing metrics
                metrics = {}
                try:
                    metrics['ss'] = silhouette_score_computing(features, res.labels_)
                    metrics['ch'] = calinski_harabaz_score_computing(features, res.labels_)
                except ValueError:
                    pdb.set_trace()

                # Computing the inertia for each cluster
                clusters_inertia = per_cluster_inertia(features, res.cluster_centers_, res.labels_)
                c_comp = clusters_composition_name(res.labels_, len(sub_data),
                                                   original_names=np.asarray([o.name for o in sub_data]),
                                                   verbose=False)

                # Appending the value for the current k to the results OrderedDict
                            # (used for graph)
                if data_to_graph and data_to_graph in metrics.keys():
                                res_k.append([k, metrics[data_to_graph]])
                # Compute the distance between each centroids for each dimension
                centroids = np.asarray(res.cluster_centers_)
                centroids_diff = {}
                for i in range(len(centroids) - 1):
                    for j in range(i+1, len(centroids)):
                        centroids_diff[f'c{i}_c{j}'] = abs(centroids[i] - centroids[j]).tolist()

                # Add the current algorithm output to the results
                results.add_values(k=k, data_used=data_to_select, joints_used=joint, centroids=res.cluster_centers_,
                                   centroids_diff=centroids_diff, global_inertia=res.inertia_,
                                   per_cluster_inertia=clusters_inertia, metrics=metrics, motion_repartition=c_comp)

            path_to_export = ''

            # If the length of the folder is too long, we shorten it

            if not os.path.exists(r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + person_name):
                os.makedirs(r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + person_name)

            folder_name = 'SUB_' + str(idx) + '_' + string_length_shortening(file_name)
            folder_name = folder_name.replace('BegMaxEndSpeed', 'BMES')
            path_to_export = r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + person_name + '/' + folder_name +'/'


            data_to_export = 'all'
            data_to_export = results.get_res(ss=[0.5, 'supeq'])

            if data_to_export:
                # Exporting the data
                results.export_data(path=path_to_export,
                                    data_to_export=data_to_export,
                                    text_export=True,
                                    json_export=True)

                # Placeholder values
                display_graph = False
                save_graph = True
                # Plotting or saving the desired values
                if display_graph or save_graph:
                    plot_save_name = ''

                    if save_graph:
                        plot_save_name = "_".join(data_to_select)

                    plot_save_name = plot_save_name.replace('BegMaxEndSpeed', 'BMES')

                    plot_data_sub_k_means(res_k, joint=joint, display=display_graph,
                                          save=save_graph, name=plot_save_name,
                                          path=path_to_export, graph_title=plot_save_name.replace('_', ' '),
                                          x_label='k value', y_label=data_to_graph)

def k_means_second_pass_all_data(file_path, result_path, file_name, person_name):
    """
        Make a second pass on sub-clusters.
    """

    # What a nice idea to also export results as json files...
    # Why does it produce a list instead of just the dic tho
    res_total = json.load(open(result_path + file_name))

    # For every good combination (ss > 0.5)
    for one_res in res_total:
        print(f'Processing for {one_res["k"]} clusters')
        print(f'Using {one_res["joints_used"]}')
        data_sub_separation = []

        # We get the different motion repartition
        for cluster in one_res['motion_repartition'].keys():
            data_sub_separation.append([name+'Char00' for name in one_res['motion_repartition'][cluster]])

        # For each same cluster motion, we do another pass of the algorithm
        # idx for export name
        for idx, subcomp in enumerate(data_sub_separation):
            for data_to_select in cst.data_types_combination:

                # Preparing some variables for the rest of the algorithm
                # It's actually more readable to have explicit variable names
                # instead of dictionnary keys
                joint = one_res['joints_used']
                if ',' in one_res['joints_used']:
                    joint = one_res['joints_used'].replace(' ', '').split(',')

                print(f'Running on {data_to_select}')

                sub_data = json_specific_import(file_path, subcomp)

                data_to_graph = 'ss'

                # This list will contains the result for each k
                res_k = []

                results = Results()

                max_k_value = 10

                if len(sub_data) <= max_k_value:
                    max_k_value = len(sub_data) - 1

                # No need to compute a k-means for a sample number = 2
                if max_k_value <= 2:
                    continue

                print(f'Running on c{idx} for k = 2 to {max_k_value}')
                for k in range(2, max_k_value + 1):

                     # We keep the joints' data we're interested in (from the motion class)
                    selected_data = joint_selection(sub_data, joint)

                    # We select the data we want and we put them in the right shape
                    # for the algorithm [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
                    features = data_selection(selected_data, data_to_select)

                    # Actual k-means
                    res = kmeans_algo(features, k=k)

                    # Computing metrics
                    metrics = {}
                    metrics['ss'] = silhouette_score_computing(features, res.labels_)
                    metrics['ch'] = calinski_harabaz_score_computing(features, res.labels_)

                    # Computing the inertia for each cluster
                    clusters_inertia = per_cluster_inertia(features, res.cluster_centers_, res.labels_)
                    c_comp = clusters_composition_name(res.labels_, len(sub_data),
                                                       original_names=np.asarray([o.name for o in sub_data]),
                                                       verbose=False)

                    # Appending the value for the current k to the results
                    # OrderedDict (used for graph)
                    if data_to_graph and data_to_graph in metrics.keys():
                                    res_k.append([k, metrics[data_to_graph]])
                    # Compute the distance between each centroids for each dimension
                    centroids = np.asarray(res.cluster_centers_)
                    centroids_diff = {}
                    for i in range(len(centroids) - 1):
                        for j in range(i+1, len(centroids)):
                            centroids_diff[f'c{i}_c{j}'] = abs(centroids[i] - centroids[j]).tolist()

                    # Add the current algorithm output to the results
                    results.add_values(k=k, data_used=data_to_select, joints_used=joint, centroids=res.cluster_centers_,
                                       centroids_diff=centroids_diff, global_inertia=res.inertia_,
                                       per_cluster_inertia=clusters_inertia, metrics=metrics, motion_repartition=c_comp)

                path_to_export = ''

                # If the length of the folder is too long, we shorten it

                if not os.path.exists(r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + person_name):
                    os.makedirs(r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + person_name)

                folder_name = file_name
                folder_name = folder_name.replace('.json', '').replace('_output', '')
                for key, value in cst.data_types_corres.items():
                    folder_name = folder_name.replace(key, value)
                folder_name = folder_name + '_c' + str(one_res['k'])

                joint_to_text = joint
                if isinstance(joint_to_text, list):
                    joint_to_text = '_'.join(joint_to_text)
                subfolder_name = joint_to_text + '_SUB_' + str(idx)
                for key, value in cst.joints_name_corres.items():
                    subfolder_name = subfolder_name.replace(key, value)

                subsubfolder_name = ', '.join(data_to_select).replace(" ", "").replace(",", "")
                for key, value in cst.data_types_corres.items():
                    subsubfolder_name = subsubfolder_name.replace(key, value)

                path_to_export = r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + person_name + '/' + folder_name +'/' + subfolder_name + '/' + subsubfolder_name + '/'

                data_to_export = 'all'
                data_to_export = results.get_res(ss=[0.5, 'supeq'])

                # if subfolder_name == 'BMESNorm':
                #     print(r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + person_name + '/' + folder_name +'/' + subfolder_name + '/')
                #     pdb.set_trace()

                if data_to_export:
                    # Exporting the data
                    results.export_data(path=path_to_export,
                                        data_to_export=data_to_export,
                                        text_export=True,
                                        json_export=True)

                    # if subfolder_name == 'BMESNorm':
                    #     print(r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/' + person_name + '/' + folder_name +'/' + subfolder_name + '/')
                    #     pdb.set_trace()
                    # Placeholder values
                    display_graph = False
                    save_graph = True
                    # Plotting or saving the desired values
                    if display_graph or save_graph:
                        plot_save_name = ''

                        if save_graph:
                            plot_save_name = "_".join(data_to_select)

                        for key, value in cst.data_types_corres.items():
                            plot_save_name = plot_save_name.replace(key, value)

                        plot_data_sub_k_means(res_k, joint=joint, display=display_graph,
                                              save=save_graph, name=plot_save_name,
                                              path=path_to_export, graph_title=plot_save_name.replace('_', ' '),
                                              x_label='k value', y_label=data_to_graph)


def plot_good_vs_student_all_data(expert_good_data, student_data, algorithm_and_parameter, datatype_joints_list, only_centroids=False):

    std_features = data_gathering(student_data, datatype_joints_list)
    exp_features = data_gathering(expert_good_data, datatype_joints_list)

    exp_centroid = get_centroid(exp_features)
    std_centroid = get_centroid(std_features)


    if len(exp_centroid) > 2:
        # Should never go here if called from " feedback " since normalisation
        # and PCA is done beforehand
        std_features = normalize(std_features)
        exp_features = normalize(exp_features)
        std_centroid = normalize(std_centroid.reshape(1, -1))[0]
        exp_centroid = normalize(exp_centroid.reshape(1, -1))[0]

        pca = PCA(n_components=2, copy=True)
        pca.fit(exp_features)

        exp_features = pca.transform(exp_features)
        exp_centroid = pca.transform(exp_centroid.reshape(1, -1))[0]
        std_features = pca.transform(std_features)
        std_centroid = pca.transform(std_centroid.reshape(1, -1))[0]

    max_dst_exp = max(max(distance.cdist([exp_centroid], exp_features, 'euclidean')))
    med_dst_exp = intra_cluster_dst(exp_centroid, exp_features)

    circle_expert = Circle(exp_centroid, max_dst_exp,
        limits={'center': exp_centroid, 'radius_max':max_dst_exp,
                'radius_med':(max_dst_exp + med_dst_exp) / 2,
                'radius_min':med_dst_exp},
        is_good=False)

    plot_all_defaults_at_once(exp_features, exp_centroid, circle_expert, std_features, std_centroid, only_centroids=only_centroids)


def good_and_bad_vs_student_all_data(expert_good_data, expert_bad_data, student_data, algorithm_and_parameter, datatype_joints_list, only_centroids=False):

    std_features = data_gathering(student_data, datatype_joints_list)
    exp_good_features = data_gathering(expert_good_data, datatype_joints_list)
    exp_bad_features = data_gathering(expert_bad_data, datatype_joints_list)

    exp_good_centroid = get_centroid(exp_good_features)
    exp_bad_centroid = get_centroid(exp_bad_features)
    std_centroid = get_centroid(std_features)

    max_dst_good_exp = max(max(distance.cdist([exp_good_centroid], exp_good_features, 'euclidean')))
    med_dst_good_exp = intra_cluster_dst(exp_good_centroid, exp_good_features)

    max_dst_bad_exp = max(max(distance.cdist([exp_bad_centroid], exp_bad_features, 'euclidean')))
    med_dst_bad_exp = intra_cluster_dst(exp_bad_centroid, exp_bad_features)

    circle_good_expert = Circle(exp_good_centroid, max_dst_good_exp,
        limits={'center': exp_good_centroid, 'radius_max':max_dst_good_exp,
                'radius_med':(max_dst_good_exp + med_dst_good_exp) / 2,
                'radius_min':med_dst_good_exp},
        is_good=False)

    circle_bad_expert = Circle(exp_bad_centroid, max_dst_bad_exp,
        limits={'center': exp_bad_centroid, 'radius_max':max_dst_bad_exp,
                'radius_med':(max_dst_bad_exp + med_dst_bad_exp) / 2,
                'radius_min':med_dst_bad_exp},
        is_good=False)

    print(f'PCA max dst good: {max_dst_good_exp}\nPCA max dst bad: {max_dst_bad_exp}')