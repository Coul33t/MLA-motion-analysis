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
from algos.kmeans_algo import kmeans_algo
from data_visualization import visualization, plot_2d
from data_processing import delta_computing


def test_mean_speed_intervals(motion_type="gimbal", joints_to_append=None):
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





def test_mean_speed_intervals_batch(size, motion_type='gimbal', joints_to_append=None):
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
    
    # Joint selection
    # List ( List (motion) [
    #                       str (name), dict (datatype) [
    #                                                    k:str (datatype name), v:dict (joints) [
    #                                                                                            k: str (joint), v:liss[double] (values)
    #                                                                                           ]
    #                                                   ]
    #                      ]
    # )
    # We gonna switch from dict to OrederedDict, to ensure that each
    # features vector has the same joints order
    selected_data = []

    datatypes = list(data[0][1].keys())
    # If joints_to_append isn't specified, we take all the joints.
    # Just to be sure that the order won't change between 2 motion dicts, 
    # we take the keys of one motion and use it as the reference for insertion order
    if not joints_to_append:
        joints_to_append = list(data[0][1][datatypes[0]].keys())

    for motion in data:
        # We keep the name
        selected_joints_motion = [motion[0], OrderedDict()]

        # For each datatype
        for datatype in datatypes:
            
            # We use an OrderedDict. Since we iterate over the joints_to_append list,
            # the order is ensured to be the same for each motion
            joints_selected = OrderedDict()

            # For each joint to append
            for joint in joints_to_append:
                joints_selected[joint] = motion[1][datatype][joint]

            selected_joints_motion[1][datatype] = joints_selected

        selected_data.append(selected_joints_motion)

    return selected_data





def data_selection(data, data_to_keep):

    features = []

    # For each motion
    for motion in data:
        motion_feat = OrderedDict()
        

        # Don't ask how it works (= nb_frames - 1 for speed)
        nb_values = len(list(motion[1][list(motion[1].keys())[0]].values())[0])

        for i in range(nb_values):
            # For each datatype
            for datatype in motion[1]:
                if datatype in data_to_keep:
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





def mean_matrix(data):
    """
        Compute the distance between each sample of the data
    """
    distance_matrix = np.zeros( (len(data), len(data)) )

    for i in range(len(data)):
        for j in range(len(data)):
            distance_matrix[i][j] = abs(data[i] - data[j])

    return distance_matrix




def test_full_batch(path, joints_to_append=None):
    original_data = json_import(path)

    selected_data = joint_selection(original_data, joints_to_append)

    data_to_select = ['Speed']
    features = data_selection(selected_data, data_to_select)

    res = kmeans_algo(features)
    print('Good classification rate : {}\nInertia: {}'.format(LOUP_rate(res.labels_), res.inertia_))





def test_full_batch_every_joint(path):
    original_data = json_import(path)

    joints_to_append = list(original_data[0][1]['Speed'].keys())

    data_to_select = ['Speed', 'Speedx', 'Speedy', 'Speedz']

    print('Data used: {}'.format(data_to_select))
    str1 = 'Joint name'
    str2 = 'Rate'
    str3 = 'Inertia'
    print(str1.ljust(30) + str2.ljust(15) + str3)
    print('----------------------------------------------------------------')

    for joint in joints_to_append:
        selected_data = joint_selection(original_data, [joint])
      
        features = data_selection(selected_data, data_to_select)

        res = kmeans_algo(features)
        str1 = '{}'.format(joint)
        str2 = '{}'.format(LOUP_rate(res.labels_))
        str3 = '{}'.format(res.inertia_)
        print(str1.ljust(30) + str2.ljust(15) + str3)

def LOUP_rate(labels):
    true_labels = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]

    diff = []
    for j, _ in enumerate(true_labels):
        diff.append(abs(true_labels[j]-labels[j]))

    return max(diff.count(0)/len(diff), diff.count(1)/len(diff))

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
                      ['LeftShoulder']

    ]

    for joints_to_append in joints_to_test:
        print(joints_to_append)
        test_full_batch(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\\', joints_to_append=joints_to_append)
        print('')

def main_all_joints():
    test_full_batch_every_joint(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\\')
    
if __name__ == '__main__':
    main_all_joints()


# [motion["RightHand"], motion["RightForeArm"], motion["RightArm"], motion["RightShoulder"],
#  motion["RightHandThumb1"], motion["RightHandThumb2"], motion["RightHandThumb3"],
#  motion["RightInHandIndex"], motion["RightHandIndex1"], motion["RightHandIndex2"],
#  motion["RightHandIndex3"], motion["RightInHandMiddle"], motion["RightHandMiddle1"],
#  motion["RightHandMiddle2"], motion["RightHandMiddle3"]]

# test_full_batch(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Batch_Test', joints_to_append=['RightForeArm', 'RightHandThumb1', 'RightHandThumb2', '"RightHandThumb3', 'RightInHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightInHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'])
