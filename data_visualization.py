# Native modules
import math
import pdb

import os

from math import (ceil, floor)

# Libraries modules
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib import colors

from scipy.signal import (savgol_filter, savgol_coeffs,
                          correlate, butter, lfilter)

from scipy.cluster.hierarchy import dendrogram


from sklearn.decomposition import PCA

# Personal modules
from tools import natural_keys

from visualisation.agglomerative_dendogram import plot_dendrogram

import constants as cst
#TODO : better data visualization function

def fast_savgol_visualisation():
    name = 'SeniowE'
    from feedback import import_data
    path = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\alldartsdescriptors\students_2\SeniowETEST'
    student_data = import_data(path, name)
    datatype_joints_list = []
    datatype_joints = {'SpeedNorm': [{'joint': 'RightHand', 'laterality': False}]}
    from data_processing import data_gathering
    std_features = data_gathering(student_data, datatype_joints)[0]

    fig = plt.figure()
    ls = fig.add_subplot(111)

    ls.set_xlabel('Time (s)')
    ls.set_ylabel('Linear speed (m/s)')

    x = np.linspace(0, len(std_features), len(std_features), endpoint=False)

    savgol_3 = savgol_filter(std_features, 21, 3)
    savgol_5 = savgol_filter(std_features, 51, 3)

    color=iter(cm.rainbow(np.linspace(0,3,len(std_features))))

    ls.plot(x, std_features, color='blue')
    ls.plot(x, savgol_5, color='red')
    ls.plot(x, savgol_3, color='green')

    current_color = next(color)
    blue_patch = mpatches.Patch(color=current_color, label='Signal original')
    red_patch = mpatches.Patch(color=current_color, label='Savgol, taille fenêtre = 51')
    green_patch = mpatches.Patch(color=current_color, label='Savgol, taille fenêtre = 21')
    plt.legend(handles=[blue_patch, green_patch, red_patch])
    plt.show()


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

#TODO: add a red thing at the best value (depending on the computed score)
def plot_data_k_means(data, display=False, save=False, name='foo', path=None, graph_title=None, x_label=None, y_label=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    color=iter(cm.rainbow(np.linspace(0,1,len(data))))

    if y_label == 'ss':
        plt.ylim((0,1))
        plt.axhline(y=0.5, color='black', linestyle='dotted', linewidth=1)

    if graph_title:
        plt.title(graph_title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    for joint in data:
        c = next(color)
        y = [k[1] for k in data[joint]]
        # clusters from 2 to 10
        x = np.linspace(2, len(y)+1, len(y))

        joint_name = joint

        joint_name = joint_name.split(',')
        for i, elem in enumerate(joint_name):
            joint_name[i] = cst.joints_name_corres[elem]
        joint_name = ', '.join(joint_name)

        ax.plot(x, y, '-', color=c, label=joint_name)

    plt.legend()

    if save:
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + name + '.png', dpi=600)
        else:
            plt.savefig(name + '.png', dpi=600)
    if display:
        plt.show()

    plt.close()

def plot_data_sub_k_means(data, joint, display=False, save=False, name='foo', path=None, graph_title=None, x_label=None, y_label=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    color=iter(cm.rainbow(np.linspace(0,1,len(data))))

    if y_label == 'ss':
        plt.ylim((0,1))
        plt.axhline(y=0.5, color='black', linestyle='dotted', linewidth=1)

    if graph_title:
        plt.title(graph_title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    y = [k[1] for k in data]
    # clusters from 2 to 10
    x = np.linspace(2, len(y)+1, len(y))

    joint_name = joint
    if not isinstance(joint_name, list):
        joint_name = [joint_name]

    for i, elem in enumerate(joint_name):
        joint_name[i] = cst.joints_name_corres[elem]
    joint_name = ', '.join(joint_name)

    ax.plot(x, y, '-', color='red', label=joint_name)

    plt.legend()

    if save:
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + name + '.png', dpi=600)
        else:
            plt.savefig(name + '.png', dpi=600)
    if display:
        plt.show()

    plt.close()


def plot_res(res_data, metrics='all', save=False, name='foo'):


    fig = plt.figure()

    rows = 2

    if len(metrics) == 1:
        rows = 1

    columns = math.ceil(len(metrics)/2)
    # rows, columns, current plot

    # PSEUDO-CODE (basically Python but whatever)
    # Actually Python lol

    # OMG MY get_res() FUNCTION IS SO AWESOME
    # THANK YOU PAST ME FOR TAKING THE TIME TO CODE A SO AWESOME FUNCTION

    # For each desired metric plot
    for i,m in enumerate(metrics):

        joint_combination = set()

        for res in res_data.results_list:
            joint_combination.add(res['joint_used'])

        ax = fig.add_subplot(rows, columns, i)
        color=iter(cm.rainbow(np.linspace(0,1,len(joint_combination))))

        for joint in joint_combination:
            c = next(color)
            y = 0 # TODO: FINISH
            # clusters from 2 to 10
            x = np.linspace(2, len(y)+1, len(y))

            ax.plot(x, y, '-', color=c, label=joint)

        plt.legend()

        if save:
            plt.savefig(name + '.png')
        else:
            plt.show()



def simple_plot_2d(data, axis_lim=None):
    data = np.asarray(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(0, data.shape[0]-1, data.shape[0])

    ax.set_xlabel('Frame')
    ax.set_ylabel('Linear speed (m/s)')

    if axis_lim:
        ax.set_xlim(axis_lim)
    ax.plot(x, data.reshape(data.shape[0]), color='blue')

    plt.show()

def simple_plot_2d_2_curves(data1, data2):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    try:
        len_total = max(data1.shape[1], data2.shape[1])
        idx_shape = 1
    except IndexError:
        len_total = max(data1.shape[0], data2.shape[0])
        idx_shape = 0

    x = np.linspace(0, len_total-1, len_total)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Linear speed (m/s)')

    if data1.shape[idx_shape] > data2.shape[idx_shape]:
        data2 = np.concatenate((data2, np.asarray([None for x in range(data1.shape[idx_shape] - data2.shape[idx_shape])])))
    else:
        data1 = np.concatenate((data1, np.asarray([None for x in range(data2.shape[idx_shape] - data1.shape[idx_shape])])))

    ax.plot(x, data1, color='blue')
    ax.plot(x, data2, color='red')


    plt.show()

def simple_plot_curves(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    max_length = 0
    processed_data = []

    # We put every curve into processed_data
    for curve in data:
        current_curve = np.asarray(curve)

        if current_curve.shape[0] > max_length:
            max_length = current_curve.shape[0]

        processed_data.append(current_curve)
    # We normalise the length of every curve (adding 0 after)

    x = np.linspace(0, max_length-1, max_length)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Linear speed (m/s)')

    color=iter(cm.rainbow(np.linspace(0,1,len(processed_data))))
    for i, curve in enumerate(processed_data):
        if curve.shape[0] < max_length:
            processed_data[i] = np.concatenate((curve, np.asarray([None for x in range(max_length - curve.shape[0])])))
        c = next(color)
        ax.plot(x, processed_data[i], color=c)

    plt.show()

def plot_PCA(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    pca = PCA(n_components=2, copy=True)
    pca_ed = pca.fit_transform(data)

    color = cm.rainbow(np.linspace(0,1,len(np.unique(labels))))

    if -1 in labels:
        labels = labels + 1

    for i, pt in enumerate(pca_ed):
        plt.plot(pt[0], pt[1], 'o', color=color[labels[i]])

    plt.show()

def multi_plot_PCA(data, labels, clusters_names, names, models, sss, title=None, trapezoids=None, circles=None, only_centroids=False, centroids=None, std_data=None):

    size = len(data)

    if trapezoids and len(trapezoids) != size:
        print(f"ERROR: not enough trapezoids.")
        return

    if circles and len(circles) != size:
        print(f"ERROR: not enough circles.")
        return

    # More for dendrogram
    if 'agglomerative' in names:
        size += names.count('agglomerative')

    # Computing the needed space for plotting
    final_x = 0

    for i in range(10):
        if pow(i,2) < size:
            final_x += 1

    final_y = ceil(size / final_x)

    fig, axs = plt.subplots(final_y, final_x, sharex=False, sharey=False)

    fig.suptitle(title, fontsize=30)

    i = 0

    # While i instead of for i because of the dendrograms
    while i < (len(data)):

        pca_ed = data[i]
        pca_ed_centroids = None
        pca_ed_std_data = None

        if centroids:
            pca_ed_centroids = centroids[i]
        if std_data:
            pca_ed_std_data = std_data[i][0]

        # A PCA is ran on the data to plot them in 2D if needed
        if data[i].shape[1] > 2:
            pca = PCA(n_components=2, copy=True)
            pca_ed = pca.fit_transform(data[i])
            if centroids:
                pca_ed_centroids = pca.fit_transform(centroids[i])
            if std_data:
                pca_ed_std_data = pca.fit_transform(std_data[i][0])


        current_labels = labels[i]

        # Let say that we take classes 2 and 3 for the test, algorithms
        # will still labels clusters from 0 to n, but ground truth will
        # label them as 2 and 3, which is problematic for index display
        if names[i] == 'Ground Truth':
            # Ground Truth is casted to a numpy array, see the assignation
            # line in the next for loop
            current_labels = np.asarray(current_labels)

            # Getting all the clusters numbers
            labels_idx = np.unique(current_labels)
            # Changing it for every cluster
            for idx_off, idx in enumerate(labels_idx):
                idx = int(idx)
                current_labels[current_labels == idx] = current_labels[current_labels == idx] - idx + idx_off

        current_axis = axs[floor(i/final_x)][i%final_x]

        color = cm.rainbow(np.linspace(0,1,len(np.unique(current_labels))))

        if trapezoids:
            patch = patches.PathPatch(trapezoids[i], facecolor='PaleGreen', lw=0)
            current_axis.add_patch(patch)

        if circles:
            if 'radius_max' in circles[i][0]:
                circle_1_1 = patches.Circle(circles[i][0]['center'], circles[i][0]['radius_max'], facecolor="DarkMagenta", lw=0)
                current_axis.add_patch(circle_1_1)
            if 'radius_med' in circles[i][0]:
                circle_1_2 = patches.Circle(circles[i][0]['center'], circles[i][0]['radius_med'], facecolor="DodgerBlue", lw=0)
                current_axis.add_patch(circle_1_2)
            if 'radius_min' in circles[i][0]:
                circle_1_3 = patches.Circle(circles[i][0]['center'], circles[i][0]['radius_min'], facecolor="SkyBlue", lw=0)
                current_axis.add_patch(circle_1_3)

            if 'radius_max' in circles[i][1]:
                circle_2_1 = patches.Circle(circles[i][1]['center'], circles[i][1]['radius_max'], facecolor="DarkMagenta", lw=0)
                current_axis.add_patch(circle_2_1)
            if 'radius_med' in circles[i][1]:
                circle_2_2 = patches.Circle(circles[i][1]['center'], circles[i][1]['radius_med'], facecolor="DodgerBlue", lw=0)
                current_axis.add_patch(circle_2_2)
            if 'radius_min' in circles[i][1]:
                circle_2_3 = patches.Circle(circles[i][1]['center'], circles[i][1]['radius_min'], facecolor="SkyBlue", lw=0)
                current_axis.add_patch(circle_2_3)

        if not only_centroids:
            for j, pt in enumerate(pca_ed):

                # if the point has -1 as a label, it means that it's been considered
                # as an outlier, so we display it in black
                current_color = color[current_labels[j]] if current_labels[j] != -1 else np.array([0, 0, 0])

                current_axis.plot(pt[0], pt[1], 'o', color=current_color, markersize=10)

                # The label of the point is added (data number)
                current_axis.annotate(j, xy=(pt[0], pt[1]), color=luminance(color[current_labels[j]]), ha='center', va='center', fontsize=7)

        else:
            for j, centroid in enumerate(pca_ed_centroids):
                current_color = color[current_labels[j]]
                current_axis.plot(centroid[0], centroid[1], 'o', color=current_color, markersize=10)
            for j, std_pt in enumerate(pca_ed_std_data):
                current_color = (0,0,0)
                current_axis.plot(std_pt[0], std_pt[1], 'o', color=current_color, markersize=10)
                current_axis.annotate(j, xy=(std_pt[0], std_pt[1]), color=luminance(current_color), ha='center', va='center', fontsize=7)



        current_axis.set_title(names[i] + ' (ss = ' + str(sss[i]) + ')')
        ax = current_axis.axis()
        min_val = min(ax[0], ax[2])
        max_val = max(ax[1], ax[3])
        current_axis.set_xlim([min_val, max_val])
        current_axis.set_ylim([min_val, max_val])


        # If it's an agglomerative clustering, we plot the dendrogram next to it
        if names[i] == 'agglomerative':
            i += 1
            current_axis = axs[floor(i/final_x)][i%final_x]
            current_axis.set_title('Previous ' + names[i-1] + ' dendrogram')
            plot_dendrogram(models[i-1], ax=current_axis)

        i += 1

    plt.show()

def plot_all_defaults(clustering_problems, only_centroids=False, title="Apprenant VS expert"):
    # Computing the needed space for plotting
    size = len(clustering_problems)

    final_x = 0

    for i in range(10):
        if pow(i,2) < size:
            final_x += 1

    final_y = ceil(size / final_x)

    fig, axs = plt.subplots(final_y, final_x, sharex=False, sharey=False)

    fig.suptitle(title, fontsize=30)

    for i, clustering_prob in enumerate(clustering_problems):
        pca_ed_data = clustering_prob.features
        pca_ed_centroids = clustering_prob.centroids
        pca_ed_std_data = clustering_prob.std_data
        pca_ed_std_centroid = clustering_prob.std_centroid

        if clustering_prob.features.shape[1] > 2:
            pca = PCA(n_components=2, copy=True)
            pca_ed_data = pca.fit_transform(clustering_prob.features)
            pca_ed_centroids = pca.fit_transform(clustering_prob.centroids)
            pca_ed_std_data = pca.fit_transform(clustering_prob.std_data)
            pca_ed_std_centroid = pca.fit_transform(clustering_prob.std_centroid)

        current_axis = axs[floor(i/final_x)][i%final_x]

        if clustering_prob.trapezoid:
            patch = patches.PathPatch(clustering_prob.trapezoid.path, facecolor='PaleGreen', lw=0)
            current_axis.add_patch(patch)

        if clustering_prob.circles:
            for i, circle in enumerate(clustering_prob.circles):
                if circle.is_good:
                    c1 = 'DarkMagenta'
                    c2 = 'DodgerBlue'
                    c3 = 'SkyBlue'
                else:
                    c1 = 'LightSalmon'
                    c2 = 'IndianRed'
                    c3 = 'Crimson'

                circle_to_draw = patches.Circle(circle.center, circle.limits['radius_max'], facecolor=c1, lw=0)
                current_axis.add_patch(circle_to_draw)
                circle_to_draw = patches.Circle(circle.center, circle.limits['radius_med'], facecolor=c2, lw=0)
                current_axis.add_patch(circle_to_draw)
                circle_to_draw = patches.Circle(circle.center, circle.limits['radius_min'], facecolor=c3, lw=0)
                current_axis.add_patch(circle_to_draw)

        c_good = 'Green'
        c_bad = 'BlueViolet'

        if not only_centroids:
            for j, pt in enumerate(pca_ed_data):

                current_color = c_good
                #if clustering_prob.labels[j]

                current_color = colors.to_rgb(current_color)

                if len(pt) == 1:
                    pt = np.asarray([pt[0], 0.0])
                current_axis.plot(pt[0], pt[1], 'o', color=current_color, markersize=10)

                # The label of the point is added (data number)

                current_axis.annotate(j, xy=(pt[0], pt[1]), color=luminance(current_color), ha='center', va='center', fontsize=7)

        else:
            for j, centroid in enumerate(pca_ed_centroids):
                current_color = c_good
                if len(centroid) == 1:
                    centroid = np.asarray([centroid[0], 0.0])
                current_axis.plot(centroid[0], centroid[1], 'o', color=current_color, markersize=10)

            for j, std_pt in enumerate(pca_ed_std_data):
                current_color = (0,0,0)
                if len(std_pt) == 1:
                    std_pt = np.asarray([std_pt[0], 0.0])
                current_axis.plot(std_pt[0], std_pt[1], 'o', color=current_color, markersize=10)
                current_axis.annotate(j, xy=(std_pt[0], std_pt[1]), color=luminance(current_color), ha='center', va='center', fontsize=7)

            if len(pca_ed_std_centroid) == 1:
                pca_ed_std_centroid = np.asarray([pca_ed_std_centroid[0], 0.0])
            current_axis.plot(pca_ed_std_centroid[0], pca_ed_std_centroid[1], 'o', color='red', markersize=15)
            current_axis.annotate('c', xy=(pca_ed_std_centroid[0], pca_ed_std_centroid[1]), color='black', ha='center', va='center', fontsize=15)



        current_axis.set_title(clustering_prob.algo_name)
        ax = current_axis.axis()
        min_val = min(ax[0], ax[2])
        max_val = max(ax[1], ax[3])
        current_axis.set_xlim([min_val, max_val])
        current_axis.set_ylim([min_val, max_val])


    plt.show()

def plot_all_defaults_at_once(exp_features, exp_centroid, exp_circle, std_features, std_centroid, only_centroids=False, title="Apprenant VS expert"):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title(title, fontsize=30)

    # Plotting expert circle
    c1 = 'DarkMagenta'
    c2 = 'DodgerBlue'
    c3 = 'SkyBlue'

    circle_to_draw = patches.Circle(exp_circle.center, exp_circle.limits['radius_max'], facecolor=c1, lw=0)
    ax.add_patch(circle_to_draw)
    circle_to_draw = patches.Circle(exp_circle.center, exp_circle.limits['radius_med'], facecolor=c2, lw=0)
    ax.add_patch(circle_to_draw)
    circle_to_draw = patches.Circle(exp_circle.center, exp_circle.limits['radius_min'], facecolor=c3, lw=0)
    ax.add_patch(circle_to_draw)

    c_good = 'Green'
    c_bad = 'BlueViolet'

    if not only_centroids:
        for j, pt in enumerate(exp_features):

            current_color = c_good
            current_color = colors.to_rgb(current_color)

            if len(pt) == 1:
                pt = np.asarray([pt[0], 0.0])
            ax.plot(pt[0], pt[1], 'o', color=current_color, markersize=10)

            # The label of the point is added (data number)
            ax.annotate(j, xy=(pt[0], pt[1]), color=luminance(current_color), ha='center', va='center', fontsize=7)

    else:
        current_color = c_good
        ax.plot(exp_centroid[0], exp_centroid[1], 'o', color=current_color, markersize=10)

        for j, std_pt in enumerate(std_features):
            current_color = (0,0,0)
            if len(std_pt) == 1:
                std_pt = np.asarray([std_pt[0], 0.0])
            ax.plot(std_pt[0], std_pt[1], 'o', color=current_color, markersize=10)
            ax.annotate(j, xy=(std_pt[0], std_pt[1]), color=luminance(current_color), ha='center', va='center', fontsize=7)

        ax.plot(std_centroid[0], std_centroid[1], 'o', color='red', markersize=15)
        ax.annotate('c', xy=(std_centroid[0], std_centroid[1]), color='black', ha='center', va='center', fontsize=15)

    plt.show()

def plot_progression(clustering_problems, title=None, text=None):
    # Computing the needed space for plotting
    size = len(clustering_problems[0])

    final_x = 0

    for i in range(10):
        if pow(i,2) < size:
            final_x += 1

    final_y = ceil(size / final_x)

    fig, axs = plt.subplots(final_y, final_x, sharex=False, sharey=False)

    fig.suptitle(title, fontsize=30)

    # clustering_prob is a list of list, 4 ClusteringProblem for each default
    for i, clustering_prob in enumerate(clustering_problems):
        # Won't change between different ClusteringProblem for each default
        pca_ed_centroids = clustering_prob[0].centroids

        if clustering_prob[0].features.shape[1] > 2:
            pca = PCA(n_components=2, copy=True)
            pca_ed_centroids = pca.fit_transform(clustering_prob[0].centroids)

        final_std_centroids = []

        for sub_cp in clustering_prob:
            pca_ed_std_centroid = sub_cp.std_centroid

            if sub_cp.features.shape[1] > 2:
                pca_ed_std_centroid = pca.fit_transform(pca_ed_std_centroid.std_centroid)

            final_std_centroids.append(pca_ed_std_centroid)

        current_axis = axs[floor(i/final_x)][i%final_x]

        if clustering_prob[0].trapezoid:
            patch = patches.PathPatch(clustering_prob[0].trapezoid.path, facecolor='PaleGreen', lw=0)
            current_axis.add_patch(patch)

        if clustering_prob[0].circles:
            for i, circle in enumerate(clustering_prob[0].circles):
                if circle.is_good:
                    c1 = 'DarkMagenta'
                    c2 = 'DodgerBlue'
                    c3 = 'SkyBlue'
                else:
                    c1 = 'LightSalmon'
                    c2 = 'IndianRed'
                    c3 = 'Crimson'

                circle_to_draw = patches.Circle(circle.center, circle.limits['radius_max'], facecolor=c1, lw=0)
                current_axis.add_patch(circle_to_draw)
                circle_to_draw = patches.Circle(circle.center, circle.limits['radius_med'], facecolor=c2, lw=0)
                current_axis.add_patch(circle_to_draw)
                circle_to_draw = patches.Circle(circle.center, circle.limits['radius_min'], facecolor=c3, lw=0)
                current_axis.add_patch(circle_to_draw)

        c_good = 'Green'
        c_bad = 'BlueViolet'

        for j, centroid in enumerate(pca_ed_centroids):
            current_color = c_good
            current_axis.plot(centroid[0], centroid[1], 'o', color=current_color, markersize=10)

        colors_for_centroids = [(1,0,0), (1,0.7,0.1), (0,0,1), (0,1,0)]
        for j, std_pt in enumerate(final_std_centroids):
            current_color = colors_for_centroids[j]
            if j < len(final_std_centroids)-1:
                current_axis.plot([std_pt[0], final_std_centroids[j+1][0]], [std_pt[1], final_std_centroids[j+1][1]], 'o-', color=current_color, markersize=15)
            current_axis.plot(std_pt[0], std_pt[1], 'o', color=current_color, markersize=15)
            current_axis.annotate(j+1, xy=(std_pt[0], std_pt[1]), color=luminance(current_color), ha='center', va='center', fontsize=13)



        current_axis.set_title(clustering_prob[0].algo_name)
        ax = current_axis.axis()
        min_val = min(ax[0], ax[2])
        max_val = max(ax[1], ax[3])
        current_axis.set_xlim([min_val, max_val])
        current_axis.set_ylim([min_val, max_val])


    plt.show()

def luminance(colour):
    """
        Compute the luminance of a colour, and return the adequate colour
        to write on the initial one (black or white).
    """
    if (0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]) > 0.5:
        return np.array([0, 0, 0])

    return np.array([1, 1, 1])

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

def test_n_curves():
    from data_import import json_import
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/testaccbugandjerk/'
    data = []
    for i in range(100):
        data.append(json_import(path, f'Leo_{i+1}Char00')[0].get_datatype('SpeedNorm').get_joint('RightHand'))

    simple_plot_curves(data)

if __name__ == '__main__':
    fast_savgol_visualisation()
    # test_n_curves()
    # from data_import import json_import
    # path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/testaccbugandjerk/'
    # yo = json_import(path, 'Leo_1Char00')
    # yo = yo[0]

    # path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/testaccbugandjerk/'
    # ya = json_import(path, 'Leo_2Char00')
    # ya = ya[0]

    # normo = yo.get_datatype('SpeedNorm')
    # normov = normo.get_joint('RightHand')

    # norma = ya.get_datatype('SpeedNorm')
    # normav = norma.get_joint('RightHand')


    # pdb.set_trace()
    # simple_plot_2d_2_curves(normov, normav)

    # from data_import import json_import
    # path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed'
    # yo = json_import(path, 'TEST_VIS')
    # yo = yo[0]

    # norm = yo.get_datatype('Norm')
    # nnorm = yo.get_datatype('SavgoledNorm')
    # snorm = yo.get_datatype('NewThrowNorm')
    # normv = norm.get_joint('LeftHand')
    # nnormv = nnorm.get_joint('LeftHand')
    # snormv = snorm.get_joint('LeftHand')

    # pdb.set_trace()
