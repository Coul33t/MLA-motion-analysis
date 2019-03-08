import numpy as np

from data_import import json_import
from tools import flatten_list

def import_data(path, import_find):
    original_data = []

    # Gathering the data
    # If it's a list, then we have to import multiple people's data
    if isinstance(import_find[0], list):
        for to_import in import_find[0]:
            original_data.extend(json_import(path, to_import))
    # Else, it's just a name, so we import their data
    else:
        original_data = json_import(path, import_find)

    if not original_data:
        print('ERROR: no data found (check your names).')
        return

    return original_data

def data_gathering(data, datatype_joints: dict):
    """
    """

    # data_joint is a dict:
    #     {'datatype'(str): ['joint1', 'joint2'](list)}

    # return is:
    # [[sample(1)], [sample(2)], ..., [sample(n)]] -> 1 sample = 1 motion
    #
    # sample is:
    # [j1f1, j1f2, j1f3, ..., j1fm,
    #  j2f1, j2f2, j2f3, ..., j2fm,
    #  ...,
    #  jnf1, jnf2, jnf3, ..., jnfm]
    #
    # return is:
    # [[j1f1, j1f2, j1f3, ..., j1fm, j2f1, j2f2, j2f3, ..., j2fm, ..., jnf1, jnf2, jnf3, ..., jnfm], -> MOTION 1
    #  [j1f1, j1f2, j1f3, ..., j1fm, j2f1, j2f2, j2f3, ..., j2fm, ..., jnf1, jnf2, jnf3, ..., jnfm]] -> MOTION 2


    # Since I'm not 100% used to numpy array, it'll be
    # converted to it right before the return
    features = []

    # for each motion, extract the same values
    for motion in data:
        tmp_features_list = []
        # for each required datatype/joints combination for the current motion:
        for datatype, joints in datatype_joints.items():
            # for each joint, get the datatypes
            for joint in joints:
                # get the joints values for a specified datatype
                tmp_features_list.append(motion.get_datatype(datatype).get_joint_values(joint))

        features.append(flatten_list(tmp_features_list))

    return features

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


def run_clustering(path, original_data, name, validate_data=False,
                   joint_to_use=None, data_to_select=None, algorithm='k-means',
                   parameters={}, true_labels=None, verbose=False, to_file=True,
                   to_json=True, return_data=False):

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
    if not data_to_select:
        data_to_select = set([name for motion in original_data for name in motion.datatypes])

    if verbose:
        print('\nData used: {}'.format(data_to_select))

    # If there's no joint to select, then we take all of them
    if joint_to_use is None:
        joint_to_use = original_data[0].get_joint_list()

    if verbose:
        print('Joints used: {}\n'.format(joint_to_use))

    # This dict will contain each joint as a key, and for each
    # joint, a list of list, [nb_cluster, inertia]
    # Since Python 3.7, no need to use OrderedDict to preserve
    # insertion order
    res_k = {}

    # Initialising
    for joint in joint_to_use:
        # If it's a combination of joints
        if isinstance(joint, list):
            res_k[','.join(joint)] = []
        else:
            res_k[joint] = []

    results = Results()

    if isinstance(joint_to_use, list):
        joint_to_use = ','.join(joint_to_use)

    # We keep the joints' data we're interested in
    # (from the motion class)
    selected_data = joint_selection(original_data, joint_to_use)

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
    if true_labels and k == len(np.unique(true_labels)):
        metrics.update(compute_all_gt_metrics(res.labels_, true_labels))

    metrics.update(compute_all_clustering_metrics(features, res.labels_))

    # Features are returned to be plotted
    if return_data:
        return(res, metrics, features)

    return (res, metrics)


if __name__ == '__main__':
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/testBB/'
    name = 'me'
    original_data = import_data(path, [name])

    breakpoint()
    datatypes_joints = {}

    joints = ['LeftArmLeftForeArmLeftHandLeftShoulder']
    data = ['BoundingBoxMinusX', 'BoundingBoxPlusX',
          'BoundingBoxMinusY', 'BoundingBoxPlusY',
          'BoundingBoxMinusZ', 'BoundingBoxPlusZ']

    for dat in data:
        datatypes_joints[dat] = joints

    data_gathering(original_data, datatypes_joints)