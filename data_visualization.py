import math
import pdb

import os

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.signal import savgol_filter, savgol_coeffs, correlate

from scipy.signal import butter, lfilter


from data_import import data_gathering_dict

from tools import natural_keys
#TODO : better data visualization function


def visualization(motion_type="gimbal", joints_to_visualize=None, savgol=False):
    folder_path_lin = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\Damien\TEST_CUT_MAX'
    # folder_path_lin = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\Damien_1_Char00\lin'

    interframe_time = 0.017

    data_lin = data_gathering_dict(folder_path_lin)

    fig = plt.figure()
    ls = fig.add_subplot(111)

    ls.set_xlabel('Time (s)')
    ls.set_ylabel('Linear speed (m/s)')


    y = []
    
    if joints_to_visualize:
        for joint in joints_to_visualize:
            y.append((joint, data_lin.get(joint)))

    else:
        for joint in data_lin.keys():
            y.append((joint, data_lin.get(joint)))

    x = np.linspace(0, len(y[0][1]), len(y[0][1]), endpoint=False)

    # ----- COLORS ------ #

    color=iter(cm.rainbow(np.linspace(0,1,len(y))))

    # YAPAT (Yet another Python awesome trick)
    # https://stackoverflow.com/questions/849369/how-do-i-enumerate-over-a-list-of-tuples-in-python

    for i, (joint_name, values) in enumerate(y):
        my_color = next(color)
        print(joint_name)
        
        if savgol:
            print(savgol_coeffs(21, 3))
            values = savgol_filter(values, 21, 3)
            
        ls.plot(x, values, color=my_color, label=joint_name)
   
    plt.show()


def multifiles_visualization(motion_type="gimbal", joints_to_visualize=None, savgol=False):
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data_lin = []
    
    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            data_lin.append(data_gathering_dict(folder+'\\'+name+'\\lin'))

    data = []

    for dic in data_lin:
        if joints_to_visualize:
            for joint in joints_to_visualize:
                data.append(dic.get(joint))
        else:
            for joint in dic.keys():
                data.append(dic.get(joint))

    fig = plt.figure()

    # ---------- Linear speed ----------
    ls = fig.add_subplot(111)

    ls.set_xlabel('Time (s)')
    ls.set_ylabel('Linear speed (m/s)')

    color=iter(cm.rainbow(np.linspace(0,1,len(data))))

    for i in range(len(data)):
        c=next(color)
        x = np.linspace(0, len(data[i]), len(data[i]), endpoint=False)
        
        if savgol:
            ls.plot(x, savgol_filter(data[i], 51, 3), linestyle='solid', color=c)
        else:
            ls.plot(x, data[i], linestyle='solid', color=c)

    plt.show()


def test():
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