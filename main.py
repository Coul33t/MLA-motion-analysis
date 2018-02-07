# Native packages
import os
from collections import OrderedDict
import statistics as stat
import pdb

# External lib packages
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Personnal packages
from tools import flatten_list, motion_dict_to_list, natural_keys, select_joint
from data_import import data_gathering_dict, return_data, adhoc_gathering, json_import
from algos.kmeans_algo import kmeans_algo, per_cluster_inertia, f_score_computing, adjusted_mutual_info_score_computing, adjusted_rand_score_computing
from data_visualization import visualization, plot_2d, plot_data_k_means
from data_processing import delta_computing

# Natural index (1 -> first element, opposed to array idx where 0 -> first element)
GLOUP_SUCCESS = [1, 8, 19, 20, 22, 24, 25, 28, 32, 33, 39, 40, 47, 56, 57, 60, 73, 74, 77, 79, 83, 84, 95, 99, 100]
DBRUN_SUCCESS = [2, 6, 10, 14, 15, 18, 21, 27, 29, 30, 44, 46, 48, 50, 59, 63, 64, 65, 69, 70, 71, 73, 74, 76, 77, 83, 86, 87, 88, 89, 90, 93, 97, 100]

# OLD AND UNUSED (TODO:delete)
def test_mean_speed_intervals(motion_type="gimbal", joints_to_append=None):
    """
        This function applies k-means (k=2) on motion data.
    """
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data = []

    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            print("Appending {}".format(name))
            data.append(flatten_list(motion_dict_to_list(data_gathering_dict(folder+'\\'+name+'\\lin_mean_10_cut', joints_to_append))))

    pdb.set_trace()

    return kmeans_algo(data)




# OLD AND UNUSED (TODO:delete)
def test_mean_speed_intervals_batch(size, motion_type='gimbal', joints_to_append=None):
    """
        This function applies k-means (k=2) on motion data.
    """
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    data_lin = [[] for x in range(size)]
    names = []

    # For each folder
    for name in subdirectories:

        # If the folder's name contain the name of the motion
        if motion_type in name:

            print(name)
            subsubdirectories = os.listdir(folder+'\\'+name)
            subsubdirectories.sort(key=natural_keys)

            i = 0
            # For each file in the folder
            for subname in subsubdirectories:
                if 'lin_mean' in subname:

                    if subname not in names:
                        names.append(subname)

                    print(subname)

                    # Append data
                    data_lin[i].append(flatten_list(motion_dict_to_list(data_gathering_dict(folder+'\\'+name+'\\' + subname, joints_to_append))))
                    i += 1

    res = []

    # Actual ML

    for i, different_cut in enumerate(data_lin):
        print('Batch : {}'.format(i))
        res.append(kmeans_algo(different_cut))
        # res.append(affinity_propagation_algo(different_cut))
        # res.append(mean_shift_algo(different_cut))

    return res





def joint_selection(data, joints_to_append):
    """
        This function returns a list of desired joints data, from a list of Motion objects.
    """

    # We gonna switch from dict to OrederedDict, to ensure that each
    # features vector has the same joints order
    selected_data = []

    datatypes = data[0].get_datatypes_names()

    # If joints_to_append isn't specified, we take all the joints.
    # Just to be sure that the order won't change between 2 motion dicts,
    # we take the keys of one motion and use it as the reference for insertion order
    if not joints_to_append:
        joints_to_append = data[0].get_joint_list()

    for motion in data:
        # We keep the name
        selected_joints_motion = [motion.name, OrderedDict()]

        # For each datatype
        for datatype in datatypes:

            # We use an OrderedDict. Since we iterate over the joints_to_append list,
            # the order is ensured to be the same for each motion
            joints_selected = OrderedDict()

            # For each joint to append
            if isinstance(joints_to_append, list):
                for joint in joints_to_append:
                    joints_selected[joint] = motion.get_datatype(datatype).get_joint_values(joint)
            else:
                joints_selected[joint] = motion.get_datatype(datatype).get_joint_values(joint)

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
    # motion[1] are the values, ordered by joints and datatypes (datatype->joints->values)
    for motion in data:
        motion_feat = OrderedDict()

        # For each data type we want to gather
        for datatype in data_to_keep:

            # Don't ask how it works (= nb_frames - 1 for speed)
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
    # TODO: change the code so it uses numpy array from the beginning
    return np.asarray(features)





def distance_matrix_computing(data):
    """
        Compute the euclidean distance between each sample of the data.
    """

    distance_matrix = np.zeros( (len(data), len(data)) )

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
    fail_idx = [x for x in range(0,100) if x not in success_indexes]
    for idx in fail_idx:
        failure_distance_matrix.append(d_m[idx])
    failure_distance_matrix = np.asarray(failure_distance_matrix)

    # Mask corresponding to the success
    success_mask = np.ones(len(d_m), dtype=bool)
    success_mask[success_indexes] = False
    # Mask corresponding to the failures
    failure_mask = np.ones(len(d_m), dtype=bool)
    failure_mask[fail_idx] = False

    cutted_success_distance_matrix = np.zeros( (len(success_distance_matrix), len(success_distance_matrix)) )
    cutted_failure_distance_matrix = np.zeros( (len(failure_distance_matrix), len(failure_distance_matrix)) )

    for i, _ in enumerate(success_distance_matrix):
        cutted_success_distance_matrix[i] = success_distance_matrix[i][failure_mask]

    for i, _ in enumerate(failure_distance_matrix):
        cutted_failure_distance_matrix[i] = failure_distance_matrix[i][success_mask]

    return [np.mean(cutted_success_distance_matrix),
            np.std(cutted_success_distance_matrix),
            np.mean(cutted_failure_distance_matrix),
            np.std(cutted_failure_distance_matrix)]




def test_full_batch(path, joints_to_append=None):
    original_data = json_import(path)

    selected_data = joint_selection(original_data, joints_to_append)

    data_to_select = ['Speed']
    features = data_selection(selected_data, data_to_select)

    res = kmeans_algo(features)
    print('Good classification rate : {}\nInertia: {}'.format(LOUP_rate(res.labels_), res.inertia_))





def test_full_batch_every_joint(path):
    """
        This function runs a k-means algorithm (k=2), on each joint.
    """
    original_data = json_import(path)

    joints_to_append = list(original_data[0][1]['Speed'].keys())

    data_to_select = ['DiffSpeed']

    print('Data used: {}'.format(data_to_select))
    str1 = 'Joint name'
    str2 = 'Rate'
    str3 = 'Inertia'
    print(str1.ljust(30) + str2.ljust(15) + str3)
    str1 = '(success mean)'
    str2 = '(success std)'
    str3 = '(failure mean)'
    str4 = '(failure std)'
    print(str1.ljust(15) + str2.ljust(15) + str3.ljust(15) + str4)
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')

    for joint in joints_to_append:
        selected_data = joint_selection(original_data, [joint])

        features = data_selection(selected_data, data_to_select)

        pdb.set_trace()

        d_m = distance_matrix_computing(features)

        c_res = compute_mean_std(d_m, GLOUP_SUCCESS, natural_indexes=True)

        res = kmeans_algo(features)

        if res.inertia_ < 50 and LOUP_rate(res.labels_) > 0.7:
            str1 = '{}'.format(joint)
            str2 = '{}'.format(LOUP_rate(res.labels_))
            str3 = '{}'.format(res.inertia_)
            print(str1.ljust(30) + str2.ljust(15) + str3)
            str1 = 's_mean: {}'.format(np.around(c_res[0], decimals=4))
            str2 = 's_std: {}'.format(np.around(c_res[1], decimals=4))
            str3 = 'f_mean: {}'.format(np.around(c_res[2], decimals=4))
            str4 = 'f_std: {}'.format(np.around(c_res[3], decimals=4))
            str5 = 'g_mean: {}'.format(np.around(np.mean(d_m), decimals=4))
            str6 = 'g_std: {}'.format(np.around(np.std(d_m), decimals=4))
            print(str1.ljust(18) + str2.ljust(18) + str3.ljust(18) + str4.ljust(18) + str5.ljust(18) + str6)
            print('----------------------------------------------------------------')


def test_full_batch_k_var(path, joint_to_use=None, verbose=False, to_file=False):
    """
        This function run a k-means algorithm with varying k values, on each joint.
    """

    # f -> output to file
    if to_file:
        output_file = open('output.txt', 'w')

    # Gathering the data
    original_data = json_import(path, ['JSON_BATCH_TEST', 'Damien'])

    for motion in original_data:
        motion.validate_motion()

    # Wich data to keep
    data_to_select = ['Speed']

    # If there's no joint to select, then we take all of them
    if joint_to_use is None:
        joints_to_append = original_data[0].get_joint_list()
    else:
        joints_to_append = joint_to_use


    print('Data used: {}'.format(data_to_select))

    if to_file:
        output_file.write('Data used: {}\n\n\n'.format(data_to_select))
        output_file.write('Joints used: {}\n\n\n'.format(joints_to_append))

    # This OrderedDict will contain each joint as a key, and for each
    # joint, a list of list, [nb_cluster, inertia]
    res_k = OrderedDict()

    # 2 sets:
    # high_interia -> list (set) of joints with inertia >= 50 at any k
    # low_inertia -> list (set) of joints with inertia < 50 at any k
    # The intersection of 2 set (at the end of the algorithm) allows us
    # to see which joints' intertia decreased to " acceptable " level
    # (define " acceptable ")
    low_inertia = set()
    high_inertia = set()

    # Initialising
    for joint in joints_to_append:
        # If it's a combination of joints
        if isinstance(joint, list):
            res_k[','.join(joint)] = []
        else:
            res_k[joint] = []

    # For each k value (2 - 10)
    for k in range(2, 11):
        print('k = {}'.format(k))

        if to_file:
            output_file.write('\nk = {}\n\n'.format(k))

        # For each joint combination
        for joint in joints_to_append:
            if verbose:
                print('Joint: {}'.format(joint))

            joint_name = joint

            if isinstance(joint, list):
                joint_name = ','.join(joint)

            # We keep the data we're interested in (from the .json file)
            selected_data = joint_selection(original_data, joint)

            # We put them in the right shape for the algorithm [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
            features = data_selection(selected_data, data_to_select)

            # Compute the euclidean distance between each sample
            # (Unused in this version)
            # d_m = distance_matrix_computing(features)
            #
            # c_res = compute_mean_std(d_m, GLOUP_SUCCESS, natural_indexes=True)

            # Actual k-means
            res = kmeans_algo(features, k=k)

            # Computing the inertia for each cluster
            clusters_inertia = per_cluster_inertia(features, res.cluster_centers_, res.labels_)

            # Computing the f1-score, adjusted mutual information score
            # and adjusted rand score, if it's a 2 clusters problem. This is
            # because the ground truth is defined for k = 2 (success or failure)
            if k == 2:
                fs = f_score_computing(res.labels_, DBRUN_SUCCESS)
                ami = adjusted_mutual_info_score_computing(res.labels_, DBRUN_SUCCESS)
                ars = adjusted_rand_score_computing(res.labels_, DBRUN_SUCCESS)


            # Checking the inertia for the sets
            if res.inertia_ < 50:
                low_inertia.add(joint_name)
            else:
                high_inertia.add(joint_name)

            if verbose:
                print("Inertia: {}".format(res.inertia_))
                clusters_composition(res.labels_, verbose=True)

            if to_file:
                output_file.write('Joint: {} '.format(joint_name))
                output_file.write("Inertia: {}".format(res.inertia_))

                if k == 2:
                    output_file.write(" / F1-score: {}".format(fs))
                    output_file.write(" / AMI: {}".format(ami))
                    output_file.write(" / ARS: {}\n".format(ars))

                clusters_composition(res.labels_, verbose=False, output_file=output_file)

            # Appending the value for the current k to the results OrderedDict
            res_k[joint_name].append([k, res.inertia_])


    print("Low inertia joints (< 50):{}\n".format(low_inertia))
    print("High inertia joints (>= 50):{}\n".format(high_inertia))
    print("Intersection:{}\n".format(low_inertia & high_inertia))

    if to_file:
        output_file.write("\n\nLow inertia joints (< 50):{}\n".format(low_inertia))
        output_file.write("High inertia joints (>= 50):{}\n".format(high_inertia))
        output_file.write("Intersection:{}\n\n\n\n\n".format(low_inertia & high_inertia))

    output_file.close()

    # Plotting the inertia values
    plot_data_k_means(res_k)


def LOUP_rate(labels):
    """
        This ad-hoc function is used to compute the good clsutering rate with GLOUP's data.
    """

    # Don't forget to -1 values to have array indexes
    true_labels = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]

    diff = []
    for j, _ in enumerate(true_labels):
        diff.append(abs(true_labels[j]-labels[j]))

    return max(diff.count(0)/len(diff), diff.count(1)/len(diff))


def clusters_composition(labels, verbose=False, output_file=None):
    """
        For each k in labels, this function returns the clusters' composition.
    """

    # Get a list of clusters number
    cluster_nb = set(labels)

    success = np.asarray(DBRUN_SUCCESS)
    # Switching from natural idx to array idx
    success = success - 1

    failure = [x for x in range(0, 100) if x not in success]

    for k in cluster_nb:
        # use " labels.index(k) " ??? -> not with numpy arrays
        sample_idx = [i for i, x in enumerate(labels) if x == k]

        # YAPAT
        # set(success) & set(sample_idx) -> gives the intersection of the 2 sets
        # (values that are in the 2 lists)
        if verbose:
            print("Success in c{}: {} / Failure in c{}: {}".format(k, len(set(success) & set(sample_idx)), k, len(set(failure) & set(sample_idx))))
        if output_file:
            output_file.write("Success in c{}: {} / Failure in c{}: {}\n".format(k, len(set(success) & set(sample_idx)), k, len(set(failure) & set(sample_idx))))




def display_res(result_list):
    for result in result_list:
        print('{} : [min: {}] [max: {}] [mean: {}]'.format(result[0], min(result[1]), max(result[1]), sum(result[1])/len(result[1])))


def main():
    # test_full_batch(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\\', joints_to_append=['Hips'])
    # # Python can't lenny face :(
    # print('( ͡° ͜ʖ ͡°)')
    joints_to_test = [None,
                      ['RightHand'],
                      ['RightForeArm'],
                      ['RightArm'],
                      ['RightShoulder'],
                      ['RightHand', 'RightArm'],
                      ['RightHand', 'RightArm', 'RightForeArm'],
                      ['RightHand', 'RightArm', 'RightForeArm', 'RightShoulder'],
                      ['RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightInHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightInHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'],
                      ['RightHand', 'RightArm', 'RightForeArm', 'RightShoulder', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightInHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightInHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'],
                      ['Hips'],
                      ['LeftFoot'],
                      ['Head'],
                      ['Spine'],
                      ['Spine', 'Spine1'],
                      ['Spine', 'Spine1', 'Spine2'],
                      ['Spine', 'Spine1', 'Spine2', 'Spine3'],
                      ['LeftHand'],
                      ['LeftForeArm'],
                      ['LeftArm'],
                      ['LeftShoulder']]

    for joints_to_append in joints_to_test:
        print(joints_to_append)
        test_full_batch(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\\', joints_to_append=joints_to_append)
        print('')

def main_all_joints():
    # Extracted from test_full_batch_k_var (inertia < 50 at one point)
    # low_inertia_joints = ['EndLeftFoot', 'LeftHandRing2', 'LeftForeArm', 'LeftHandPinky2', 'LeftHandMiddle2', 'EndHead', 'LeftHandIndex1', 'Spine3', 'LeftHandThumb3', 'LeftHandMiddle1', 'EndLeftHandRing3', 'LeftHandRing3', 'LeftHandMiddle3', 'RightShoulder', 'RightArm', 'LeftInHandRing', 'EndLeftHandThumb3', 'Spine', 'LeftInHandPinky', 'LeftHandPinky3', 'EndLeftHandMiddle3', 'LeftHandIndex2', 'LeftHandRing1', 'LeftHandThumb2', 'LeftShoulder', 'Hips', 'LeftHandIndex3', 'Spine2', 'EndLeftHandPinky3', 'EndLeftHandIndex3', 'LeftInHandIndex', 'RightForeArm', 'RightUpLeg', 'RightFoot', 'LeftArm', 'LeftUpLeg', 'RightLeg', 'LeftHandPinky1', 'Neck', 'LeftFoot', 'LeftInHandMiddle', 'LeftHandThumb1', 'LeftHand', 'LeftLeg', 'Spine1', 'Head', 'EndRightFoot']
    right_joints_list = ['RightHand', 'RightForeArm', 'RightArm', 'RightShoulder', 'Neck', 'Hips']
    left_joints_list = [['LeftHand', 'LeftForeArm', 'LeftArm', 'LeftShoulder', 'Neck', 'Hips'], ['RightHand']]
    test_full_batch_k_var(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\\', joint_to_use=left_joints_list, verbose=False, to_file=True)
    # test_full_batch_every_joint(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\\')

if __name__ == '__main__':
    main_all_joints()


# [motion["RightHand"], motion["RightForeArm"], motion["RightArm"], motion["RightShoulder"],
#  motion["RightHandThumb1"], motion["RightHandThumb2"], motion["RightHandThumb3"],
#  motion["RightInHandIndex"], motion["RightHandIndex1"], motion["RightHandIndex2"],
#  motion["RightHandIndex3"], motion["RightInHandMiddle"], motion["RightHandMiddle1"],
#  motion["RightHandMiddle2"], motion["RightHandMiddle3"]]

# test_full_batch(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Batch_Test', joints_to_append=['RightForeArm', 'RightHandThumb1', 'RightHandThumb2', '"RightHandThumb3', 'RightInHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightInHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'])
