import math
import pdb

import os

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.signal import savgol_filter, correlate

from data_import import data_gathering_dict
#TODO : better data visualization function
    

def visualization(motion_type="gimbal"):
    folder_path_lin = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\throw_5_gimbal_smooth_16\lin'

    interframe_time = 0.017

    data_lin = data_gathering_dict(folder_path_lin)

    fig = plt.figure()
    ls = fig.add_subplot(211)

    ls.set_xlabel('Time (s)')
    ls.set_ylabel('Linear speed (m/s)')

    y = data_lin.get('Hips')
    y2 = data_lin.get('RightHand')
    y3 = data_lin.get('RightShoulder')
    y4 = data_lin.get('RightForeArm')
    y5 = data_lin.get('RightArm')

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