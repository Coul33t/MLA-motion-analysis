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


def plot_2d(data, true_class, clu_class, label1='NONE_1'):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i,point in enumerate(data):
        
        # class 0, OK
        if clu_class[i] == 0 and true_class[i] == 0:
            ax.plot(point[0], point[1], 'x', color='green')
        # class 1, OK
        elif clu_class[i] == 1 and true_class[i] == 1:
            ax.plot(point[0], point[1], 'x', color='blue')
        # missclassed as class 0
        elif clu_class[i] == 0 and true_class[i] == 1:
            ax.plot(point[0], point[1], 'x', color='orange')
        # missclassed as class 1
        elif clu_class[i] == 1 and true_class[i] == 0:
            ax.plot(point[0], point[1], 'x', color='red')

    green_patch = mpatches.Patch(color='green', label='class 0, OK')
    blue_patch = mpatches.Patch(color='blue', label='class 1, OK')
    orange_patch = mpatches.Patch(color='orange', label='missclassed as class 0')
    red_patch = mpatches.Patch(color='red', label='missclassed as class 1')
    plt.legend(handles=[green_patch, blue_patch, orange_patch, red_patch])
    ax.set_title(label1)      
   
    plt.show()

def plot_2d_dual(data, true_class, clu_class, label1='NONE_1', label2='NONE_2'):
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for i,point in enumerate(data):
        
        if clu_class[i] == 0:
            ax.plot(point[0], point[1], 'x', color='red')
        else:
            ax.plot(point[0], point[1], 'x', color='blue')

        if true_class[i] == 0:
            ax2.plot(point[0], point[1], 'x', color='red')
        else:
            ax2.plot(point[0], point[1], 'x', color='blue')

    ax.set_title(label1)
    ax2.set_title(label2)        
   
    plt.show()

def plot_data_k_means(data, save=False, name='foo'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    color=iter(cm.rainbow(np.linspace(0,1,len(data))))

    for joint in data:
        c = next(color)
        y = [k[1] for k in data[joint]]
        # clusters from 2 to 10
        x = np.linspace(2, len(y)+1, len(y))

        ax.plot(x, y, '-', color=c, label=joint)

    plt.legend()

    if save:
        plt.savefig(name + '.png')
    else:
        plt.show()

def simple_plot_2d(data):
    data = np.asarray(data)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(0, data.shape[0]-1, data.shape[0])
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Linear speed (m/s)')


    ax.plot(x, data.reshape(data.shape[0]), color='blue')

    plt.show()

def simple_plot_2d_2_curves(data1, data2):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(0, data1.shape[0]-1, data1.shape[0])
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Linear speed (m/s)')

    ax.plot(x, data1.reshape(data1.shape[0]), color='blue')
    ax.plot(x, data2.reshape(data2.shape[0]), color='red')

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