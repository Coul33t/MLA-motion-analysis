import os
import statistics as stat

import pdb

from data_visualization import multifiles_visualization
from tools import flatten_list, motion_dict_to_list, natural_keys, select_joint
from data_import import data_gathering_dict, return_data, adhoc_gathering
from algos.kmeans_algo import kmeans_algo
from data_visualization import visualization


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





def test_full_batch(path, joints_to_append=None):
    # adhoc because for the moment, we have multiple segments
    data = adhoc_gathering(path, joints_to_append)

    data_right_hand = []

    # Extracting right hand values
    for motion in data:
        data_right_hand.append(motion['RightHand'])

    features = []

    for motion in data_right_hand:
        features.append([stat.mean(motion), max(motion)])

    res = kmeans_algo(features)
    pdb.set_trace()


def display_res(result_list):
    for result in result_list:
        print('{} : [min: {}] [max: {}] [mean: {}]'.format(result[0], min(result[1]), max(result[1]), sum(result[1])/len(result[1])))





def main():
    # visualization(joints_to_visualize=['LeftShoulder', 'LeftForeArm', 'LeftArm', 'LeftHand'], savgol=True)
    # visualization(joints_to_visualize=['LeftHand'], savgol=False)
    # multifiles_visualization(joints_to_visualize=['Head', 'LeftShoulder', 'LeftForeArm', 'LeftArm', 'LeftHand'])
    # mean_speed()
    
    # # Python can't lenny face :(
    # print('( ͡° ͜ʖ ͡°)')
    
    # test_normalization()
    # test_dtw()
    # kmeans_algo(new_data)
    # ml()
    # res = test_mean_speed_intervals("cut", joints_to_append=['Head', 'LeftShoulder', 'LeftForeArm', 'LeftArm', 'LeftHand'])
    # pdb.set_trace()
    # display_res(res)
    # test_mean_speed_intervals_batch(19, motion_type='cut')

    test_full_batch(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Test_Python')

if __name__ == '__main__':
    main()