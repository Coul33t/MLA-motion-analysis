import os
import math
import pdb
import re

from collections import OrderedDict

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from statistics import median

from scipy.signal import savgol_filter, correlate

import numpy as np

import sklearn.cluster as cl

from fastdtw import fastdtw

# DONE / Watch 110th frame (motion 10, angular speed)
# DONE / Mean speed for all joints

# Demander à Pierre de réserver les salles TD les mardis matin
# DONE / Liste propriété à relever/récupérer

# NOPE / speeds, not positions (ressortir les clusters en mouvement (centroid to motion))
# mail Nico conf ML



# NOPE / mouvement brut
# TODO / angular speed " offset "
# DONE / test du protocole avec des doctorants
# DONE / questionnaire complet (pré et post)
# DONE / Cut frames with Motion Builder
# DONE / Algo recalcul de n valeurs (diviser mouvement en n parties, calcul de vitesse moyenne sur chaque partie)
# DONE / Profiling sur l'export des data
# TODO / Même cobaye : si clustering (2) réussi, centroide = moyenne ?

# DONE / dtw


# DONE / Envoyer mail à Olivier pour le matos
# DONE / envoyer le protocole
# TODO / Trouver et tester des propriétés (ex : seulement le bras, ou alors le bras la main, la distance de la bouteille)
# TODO / Faire des tests avec d'autres gens
# TODO / Tester gant avec des petites mains
# TODO / Diapo expliquer clustering de A à Z (petit cours, en gros) (puis transiter sur notre cas des mouvements)

def bvh_parser(folder):
    pass

def file_name_gathering(folder_path):
    """
        Collects the file name from folder_path.
    """
    return  next(os.walk(folder_path))[2]

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def flatten_list(list_to_flatten):
    return [item for sublist in list_to_flatten for item in sublist]

def dic_to_list(data_dict):
    return [list(item.values()) for item in data_dict]

def motion_dict_to_list(data_dict):
    """
        Take a dict {'val1': (1,10,100), 'val2': (2,20,200), 'val3': (3,30,300)}
        and transform it into a list [(1,2,3), (10,20,30), (100,200,300)].

        Used to transform the dictionnary extracted from csv files
        into a coherent, processable data structure.
    """
    llist = list(data_dict.values())
    new_list = []

    for i in range(len(llist[0])):
        new_list.append([item[i] for item in llist])

    return new_list




def data_gathering_dict(folder_path, joints_to_append=None):
    """
        Put the data from multiples files (in the folder folder_path) into a dictionnary.
    """
    file_list = file_name_gathering(folder_path)
    # Oh my god I'm so ashamed it took me so long to find this ...
    file_list.sort(key=natural_keys)

    data = OrderedDict()

    # For each file in the folder
    for filename in file_list:

        # Open the file
        with open(folder_path+'\\'+filename, 'r') as file_data:
            text = file_data.read()

            # For each line
            for line in enumerate(text.split('\n')):
                splitted_line = line[1].split(',')

                # if there's actually something (IF not_empty AND first_value AND second_value)
                if splitted_line and splitted_line[0] and splitted_line[1]:

                    # If it's alreayd into the ordered dict, append
                    if splitted_line[0] in data:
                        pdb.set_trace()
                        if 'ang' in filename:
                            try:
                                data[splitted_line[0]].append(math.degrees(float(splitted_line[1])))
                            except:
                                pdb.set_trace()
                                print('Error occured while appending angular speed value.')             
                        else:
                            try:
                                data[splitted_line[0]].append(float(splitted_line[1]))
                            except:
                                print('Error occured while appending linear speed value.')
                            
                    # Else, add a new value
                    else:
                        if 'ang' in filename:
                            try:
                                data[splitted_line[0]] = [math.degrees(float(splitted_line[1]))]
                            except:
                                print('Error while adding angular speed value.')
                        else:
                            try:
                                data[splitted_line[0]] = [float(splitted_line[1])]
                            except:
                                print('Error while adding linear speed value.')

    return data




def delta_computing(data_dic):
    delta_dict = []

    for dic in data_dic:
        new_dic = OrderedDict()
        
        for key in dic.keys():
            delta_values = []

            for i,values in enumerate(dic[key][:-1]):
                delta_values.append(dic[key][i+1] - dic[key][i])

            new_dic[key] = delta_values#[min(delta_values), max(delta_values)]

        delta_dict.append(new_dic)

    return delta_dict


def mean_speed(data_dic):
    mean_dict = []

    for dic in data_dic:
        new_dic = OrderedDict()
        for key in dic.keys():
            new_dic[key] = sum(dic.get(key))/len(dic.get(key))
        mean_dict.append(new_dic)
    
    return mean_dict 
    


def visualization(motion_type="gimbal"):
    folder_path_lin = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\throw_5_gimbal_smooth_16\lin'
    folder_path_ang = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\throw_10_gimbal_smooth_16\ang'

    interframe_time = 0.017

    data_lin = data_gathering_dict(folder_path_lin)
    data_ang = data_gathering_dict(folder_path_ang)



    fig = plt.figure()

    # ---------- Linear speed ----------
    ls = fig.add_subplot(211)

    ls.set_xlabel('Time (s)')
    ls.set_ylabel('Linear speed (m/s)')

    y = data_lin.get('Hips')
    y2 = data_lin.get('RightHand')
    y3 = data_lin.get('RightShoulder')
    y4 = data_lin.get('RightForeArm')
    y5 = data_lin.get('RightArm')

    #yo(y2)

    x = np.linspace(0, len(y), len(y), endpoint=False)

    blue_patch = mpatches.Patch(color='blue', label='Hips')
    red_patch = mpatches.Patch(color='red', label='RightHand')
    green_patch = mpatches.Patch(color='green', label='RightShoulder')
    orange_patch = mpatches.Patch(color='orange', label='RightForeArm')
    violet_patch = mpatches.Patch(color='violet', label='RightArm')

    ls.legend(handles=[blue_patch, red_patch, green_patch, orange_patch, violet_patch])

    ls.plot(x, y, linestyle='solid', color='blue')
    ls.plot(x, y2, linestyle='solid', color='red')
    ls.plot(x, y3, linestyle='solid', color='green')
    ls.plot(x, y4, linestyle='solid', color='orange')
    ls.plot(x, y5, linestyle='solid', color='violet')

    
    # ---------- Angular speed ----------
    angs = fig.add_subplot(212)

    angs.set_xlabel('Time (s)')
    angs.set_ylabel('Angular speed (deg/s)')

    y_ang = data_ang.get('Hips')
    y2_ang = data_ang.get('RightHand')
    y3_ang = data_ang.get('RightShoulder')
    y4_ang = data_ang.get('RightForeArm')
    y5_ang = data_ang.get('RightArm')

    x_ang = np.linspace(0, len(y_ang), len(y_ang), endpoint=False)

    blue_patch = mpatches.Patch(color='blue', label='Hips')
    red_patch = mpatches.Patch(color='red', label='RightHand')
    green_patch = mpatches.Patch(color='green', label='RightShoulder')
    orange_patch = mpatches.Patch(color='orange', label='RightForeArm')
    violet_patch = mpatches.Patch(color='violet', label='RightArm')
    
    angs.legend(handles=[blue_patch, red_patch, green_patch, orange_patch, violet_patch])

    angs.plot(x_ang, y_ang, linestyle='solid', color='blue')
    angs.plot(x_ang, y2_ang, linestyle='solid', color='red')
    angs.plot(x_ang, y3_ang, linestyle='solid', color='green')
    angs.plot(x_ang, y4_ang, linestyle='solid', color='orange')
    angs.plot(x_ang, y5_ang, linestyle='solid', color='violet')
   
    plt.show()


def multifiles_visualization(motion_type="gimbal"):
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data_lin = []
    data_ang = []
    
    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            data_lin.append(data_gathering_dict(folder+'\\'+name+'\\lin'))

    data = []
    mean_len = []

    for dic in data_lin:
        mean_len.append(len(dic.get('RightForeArm')))
    
    mean_len = int(sum(mean_len)/len(mean_len))

    # 10 because artefact in the 10th first frames (approximately)
    # mean_len/4 to avoid
    for dic in data_lin:
        data.append(dic.get('RightForeArm')[int(mean_len/4):int(mean_len+mean_len/4)])

    #TODO: shift values between 0 and 1
    corr_mat = []
    for i,signal in enumerate(data):
        corr_mat.append([])
        for signal2 in data:
            corr_mat[i].append(np.argmax(correlate(signal, signal2)))

    mini = min([item for sublist in corr_mat for item in sublist])
    maxi = max([item for sublist in corr_mat for item in sublist])

    factor = (mini+maxi)/(maxi*maxi)

    for i,row in enumerate(corr_mat):
        for j,signal in enumerate(row):
            corr_mat[i][j] = corr_mat[i][j]*factor    

    print('TEST : {}'.format(np.argmax(correlate(data[1], data[1]))))
    pdb.set_trace()

    fig = plt.figure()

    # ---------- Linear speed ----------
    ls = fig.add_subplot(111)

    ls.set_xlabel('Time (s)')
    ls.set_ylabel('Linear speed (m/s)')

    color=iter(cm.rainbow(np.linspace(0,1,len(data))))

    patches = []

    for i in range(len(data)):
        c=next(color)
        patches.append(mpatches.Patch(color=c, label=i))
        x = np.linspace(0, len(data[i]), len(data[i]), endpoint=False)
        ysavgol = savgol_filter(data[i], 51, 3)
        ls.plot(x, ysavgol, linestyle='solid', color=c)

    plt.show()

    


def kmeans_algo(data, verbose=False):
    # COMPUTING
    algo_accuracy = []


    # Kmeans
    init = ['kmeans++', 'random', 'ndarray']

    for algo in init:
        
        accuracy = []
        
        if verbose:
            print("\n\n{}\n".format(algo))

        for i in range(10):
            
            if verbose: 
                print('Iteration {} ... '.format(i+1), end=' ')

            res = cl.KMeans(2, 'k-means++', n_init=20, max_iter=1000).fit(list(data))

            true_labels = [0,0,1,0,0,     0,0,0,0,1,      1,0,0,0,0,      0,1,1,0,1]

            diff = []
            for i,_ in enumerate(true_labels):
                diff.append(abs(true_labels[i]-res.labels_[i]))

            accuracy.append(max(diff.count(0)/len(diff), diff.count(1)/len(diff)))
            
            if verbose:
                print('Done.')

        algo_accuracy.append([algo, accuracy])

    if verbose:
        for i,algo in enumerate(init):
            print("Algorithm        : {}".format(algo))
            print("Accuracy         : {}".format(algo_accuracy[i][1]))
            print("Highest accuracy : {}".format(max(algo_accuracy[i][1])))
            print("Lowest accuracy  : {}".format(min(algo_accuracy[i][1])))
            print("Mean accuracy    : {}".format(sum(algo_accuracy[i][1])/len(algo_accuracy[i][1])))
            print("Median accuracy  : {}".format(median(algo_accuracy[i][1])))
            print('\n\n\n')

    return ([algo_accuracy[i] for i in range(len(init))])




def affinity_propagation_algo(data, verbose=False):
    # COMPUTING
    algo_accuracy = []


    # Which affinity to use
    algo = 'affinity'
        
    accuracy = []
    
    if verbose:
        print("\n\n{}\n".format(algo))            

    res = cl.AffinityPropagation().fit(list(data))

    true_labels = [0,0,1,0,0,     0,0,0,0,1,      1,0,0,0,0,      0,1,1,0,1]

    diff = []
    for i,_ in enumerate(true_labels):
        diff.append(abs(true_labels[i]-res.labels_[i]))

    accuracy.append(max(diff.count(0)/len(diff), diff.count(1)/len(diff)))
    
    if verbose:
        print('Done.')

    algo_accuracy = [algo, accuracy]

    if verbose:
        print("Algorithm        : {}".format(algo))
        print("Accuracy         : {}".format(algo_accuracy[i][1]))
        print("Highest accuracy : {}".format(max(algo_accuracy[i][1])))
        print("Lowest accuracy  : {}".format(min(algo_accuracy[i][1])))
        print("Mean accuracy    : {}".format(sum(algo_accuracy[i][1])/len(algo_accuracy[i][1])))
        print("Median accuracy  : {}".format(median(algo_accuracy[i][1])))
        print('\n\n\n')

    return (algo_accuracy)




def mean_shift_algo(data, verbose=False):
    # COMPUTING
    algo_accuracy = []


    # Which affinity to use
    algo = 'Mean Shift'
        
    accuracy = []
    
    if verbose:
        print("\n\n{}\n".format(algo))            

    res = cl.MeanShift().fit(list(data))

    true_labels = [0,0,1,0,0,     0,0,0,0,1,      1,0,0,0,0,      0,1,1,0,1]

    diff = []
    for i,_ in enumerate(true_labels):
        diff.append(abs(true_labels[i]-res.labels_[i]))

    accuracy.append(max(diff.count(0)/len(diff), diff.count(1)/len(diff)))
    
    if verbose:
        print('Done.')

    algo_accuracy = [algo, accuracy]

    if verbose:
        print("Algorithm        : {}".format(algo))
        print("Accuracy         : {}".format(algo_accuracy[i][1]))
        print("Highest accuracy : {}".format(max(algo_accuracy[i][1])))
        print("Lowest accuracy  : {}".format(min(algo_accuracy[i][1])))
        print("Mean accuracy    : {}".format(sum(algo_accuracy[i][1])/len(algo_accuracy[i][1])))
        print("Median accuracy  : {}".format(median(algo_accuracy[i][1])))
        print('\n\n\n')

    return (algo_accuracy)





def ml(motion_type="gimbal"):
    # DATA GATHERING
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data_lin = []
    data_ang = []
    
    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            print("Appending {}".format(name))
            data_lin.append(data_gathering_list(folder+'\\'+name+'\\lin'))

    # (Bad) NORMALIZATION
    min_size = min(map(len, data_lin))

    for i,elem in enumerate(data_lin):
        data_lin[i] = data_lin[i][10:min_size]

    #for name in subdirectories:
        #data_ang.append(data_gathering_list(folder+'\\'+name+'\\ang'))
        
    ## (Bad) NORMALIZATION
    #min_size = min(map(len, data_ang))

    #for i,elem in enumerate(data_ang):
        #data_ang[i] = data_ang[i][10:min_size]
    
    # for mo in data_lin:
    #     delta_computing(mo)

    kmeans_algo(data_lin)
    #kmeans_algo(data_ang)


def test_normalization(motion_type="gimbal"):
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data_lin = []
    data_lin_2 = []

    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            print("Appending {}".format(name))
            data_lin.append(data_gathering_dict(folder+'\\'+name+'\\lin'))
    
    # data_lin : list of dict (1 for each motion)
    # data_lin[0] : dict with joints keys (1 for each joint)
    # data_lin[0]['Hips'] : values of Hips data for the 1 motion (as much as frame number)
    

    data = []
    mean_len = []

    for dic in data_lin:
        mean_len.append(len(dic.get('Hips')))
    
    mean_len = int(sum(mean_len)/len(mean_len))

    # 10 because artefact in the 10th first frames (approximately)
    # mean_len/4 to avoid

    # (Bad) NORMALIZATION
    #min_size = min(map(len, data_lin))

    for i,_ in enumerate(data_lin):
        data_lin[i] = data_lin[i][mean_len/4:mean_len + (mean_len / 4)]

    mean_speed_data = mean_speed(data_lin)

    delta_speed_data = delta_computing(data_lin)

    new_data = []

    for i, _ in enumerate(mean_speed_data):
        tmp_list = []
        for key in data_lin[0].keys():
            tmp_list.append(mean_speed_data[i][key])
            tmp_list.extend(delta_speed_data[i][key])
        new_data.append(tmp_list)

def test_dtw(motion_type="gimbal"):
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data_lin = []

    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            print("Appending {}".format(name))
            data_lin.append(flatten_list(motion_dict_to_list(data_gathering_dict(folder+'\\'+name+'\\lin'))))
    
    # data_lin : list of dict (1 for each motion)
    # data_lin[0] : dict with joints keys (1 for each joint)
    # data_lin[0]['Hips'] : values of Hips data for the 1 motion (as much as frame number)
    
    pdb.set_trace()

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
        #res.append(kmeans_algo(different_cut))
        #res.append(affinity_propagation_algo(different_cut))
        res.append(mean_shift_algo(different_cut))
        

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
    # visualization()
    # multifiles_visualization()
    # mean_speed()
    
    # # Python can't lenny face :(
    # print('( ͡° ͜ʖ ͡°)')
    # test_normalization()
    # test_dtw()
    # kmeans_algo(new_data)
    # ml()
    # test_mean_speed_intervals("cut")
    test_mean_speed_intervals_batch(19, motion_type='cut')

if __name__ == '__main__':
    main()



# def data_gathering_list(folder_path):
# """
#     Put the data into a list.
# """
# file_list = file_name_gathering(folder_path)


# data = list()

# for filename in file_list:
#     with open(folder_path+'\\'+filename, 'r') as file_data:
#         text = file_data.read()

#         for line in enumerate(text.split('\n')):
#             splitted_line = line[1].split(',')

#             if splitted_line and splitted_line[0] and splitted_line[1]:     
#                 if 'ang' in filename:
#                     try:
#                         data.append(math.degrees(float(splitted_line[1])))
#                     except:
#                         print('Error while appending angular speed value.')
#                 else:
#                     try:
#                         data.append(float(splitted_line[1]))
#                     except:
#                         print('Error while appending linear speed value.')

# return data