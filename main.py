import os
import statistics as stat

import pdb

from sklearn.decomposition import PCA

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
    original_data = adhoc_gathering(path, joints_to_append)

    selected_data = []

    # Extracting right hand values
    for motion in original_data:
        selected_data.append([motion['RightHand'], motion['RightArm'], motion['RightForeArm'], motion['RightShoulder']])

    features = []
    # ----- MEAN AND MAX ------
    # Here, we transform the features into mean and max speed (2 features)
    
    # for motion in selected_data:
    #     feat_to_add = []
    #     for joint in motion:
    #         feat_to_add.append([stat.mean(joint), max(joint)])
    #     features.append(flatten_list(feat_to_add))

    # res = kmeans_algo(features)
    # print('{}'.format(LOUP_rate(res.labels_)))

    # ----- MEAN AND MAX ------

    # ----- FULL DATA -----
    # for motion in selected_data:
    #     features.append(flatten_list(motion))
    # res = kmeans_algo(features)
    # print('{}'.format(LOUP_rate(res.labels_)))

    # ----- FULL DATA -----

    # ----- PCA TESTS -----
    # rates = []
    # for i in range(len(flatten_list(selected_data[0])) - 1):
        
    #     features = []
    #     for motion in selected_data:
    #         features.append(flatten_list(motion))

    #     pca = PCA(n_components=i+1)
    #     pca.fit(features)
    #     features = pca.transform(features)

    #     res = kmeans_algo(features)
        
    #     rates.append((LOUP_rate(res.labels_), i+1))
    #     # print('nb features: {} / {}'.format(i+1, rate))

    # print('Best: {} ({} features)'.format(max(rates, key=lambda item:item[0])[0], max(rates, key=lambda item:item[0])[1]))
    # ----- PCA TESTS -----
   
def LOUP_rate(labels):
    true_labels = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]

    diff = []
    for j,_ in enumerate(true_labels):
        diff.append(abs(true_labels[j]-labels[j]))

    return max(diff.count(0)/len(diff), diff.count(1)/len(diff))

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

    test_full_batch(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Batch_Test')

if __name__ == '__main__':
    main()

