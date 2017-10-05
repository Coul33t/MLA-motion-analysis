import os
import pdb

from data_visualization import multifiles_visualization
from tools import flatten_list, motion_dict_to_list, natural_keys
from data_import import data_gathering_dict
from algos.kmeans_algo import kmeans_algo
from data_visualization import visualization

def test_mean_speed_intervals(motion_type="gimbal"):
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data_lin = []
    data_lin_2 = []

    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            print("Appending {}".format(name))
            data_lin.append(flatten_list(motion_dict_to_list(data_gathering_dict(folder+'\\'+name+'\\lin_mean_10_cut'))))

    pdb.set_trace()
    kmeans_algo(data_lin)





def test_mean_speed_intervals_batch(size, motion_type='gimbal'):
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
                    data_lin[i].append(flatten_list(motion_dict_to_list(data_gathering_dict(folder+'\\'+name+'\\' + subname))))
                    i += 1

    res = []

    # Actual ML

    for i, different_cut in enumerate(data_lin):
        print('Batch : {}'.format(i))
        res.append(kmeans_algo(different_cut))
        #res.append(affinity_propagation_algo(different_cut))
        #res.append(mean_shift_algo(different_cut))
        

    #display_res(res, names)
    pdb.set_trace()





def display_res(res, names):
    for i,batch in enumerate(res):
        print('\n-------------------------------')
        print('New batch : {}'.format(names[i]))
        print('-------------------------------')
        for algo in batch:
            pdb.set_trace()
            print('{} : [min: {}] [max: {}] [mean: {}]'.format(algo[0], min(algo[1]), max(algo[1]), sum(algo[1])/len(algo[1])))





def main():
    visualization()
    # multifiles_visualization()
    # mean_speed()
    
    # # Python can't lenny face :(
    # print('( ͡° ͜ʖ ͡°)')
    
    # test_normalization()
    # test_dtw()
    # kmeans_algo(new_data)
    # ml()
    # test_mean_speed_intervals("cut")
    # test_mean_speed_intervals_batch(19, motion_type='cut')

if __name__ == '__main__':
    main()