from PyQt5 import QtCore, QtWidgets, QtGui

from data_import import *
from tools import Person
from shutil import rmtree
from distutils.dir_util import copy_tree

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, MinMaxScaler, RobustScaler, Normalizer

from data_import import json_import

from data_processing import (run_clustering,
                             plot_good_vs_student_all_data,
                             good_and_bad_vs_student_all_data)

from data_selection import data_gathering

from data_visualization import (plot_PCA,
                                multi_plot_PCA,
                                plot_all_defaults)

from feedback_tools import *

from tools import Person, merge_list_of_dictionnaries

import constants as cst

from constants import problemes_et_solutions as problems_and_advices

@dataclass
class Problem :
    name : str
    caracs : list
    laterality : bool

def check_path_validity(path) :
    if path.split('/')[-1] == 'mixed':
        return True
    else :
        return False
        
def get_motion(path, import_find) :
    original_data = json_import(path, import_find)
    # Gathering the data
    # If it's a list, then we have to import multiple people's data
    if isinstance(import_find[0], list):
        for to_import in import_find[0]:
            original_data.extend(json_import(path, to_import))
    # Else, it's just a name, so we import their data
    else:
        original_data = json_import(path, import_find)

    if not original_data:
        print('ERROR: no data found (check your names).')
        return
    
    return original_data[0]

def get_joint_list(path, import_find, datatype) :
    original_data = get_motion(path, import_find)
    return original_data.datatypes[datatype].get_joint_list()

def get_datatypes_names(path, import_find) :
    original_data = get_motion(path, import_find)
    return original_data.get_datatypes_names()

def import_data(path, import_find):
    original_data = []

    # Gathering the data
    # If it's a list, then we have to import multiple people's data
    if isinstance(import_find[0], list):
        for to_import in import_find[0]:
            original_data.extend(json_import(path, to_import))
    # Else, it's just a name, so we import their data
    else:
        original_data = json_import(path, import_find)

    if not original_data:
        print('ERROR: no data found (check your names).')
        return

    return original_data

def only_feedback_new_descriptors(expert, student, path, datatype_joints_list, expert_data_repartion, param, end_msg):

    display = param['display']

    expert_data_repartion['good'] = [x+1 for x in range(10)]
    expert_data_repartion['leaning'] = [x+1 for x in range(10, 19)]
    expert_data_repartion['javelin'] = [x+1 for x in range(20, 30)]
    expert_data_repartion['align_arm'] = [x+1 for x in range(30, 40)]
    expert_data_repartion['elbow_move'] = [x+1 for x in range(40, 50)]

    folders_path = path
    if folders_path.split('/')[-1] == 'mixed':
        folders_path = "/".join(path.split('/')[:-1])

    # if not take_last_data(folders_path, student, expert, number=9):
    #     print(f'ERROR: {folders_path} does not exists')
    #     return

    # Expert Data
    expert_data = import_data(path, expert.name)
    # Student data
    student_data = import_data(path, student.name)

    # Setting the laterality
    for motion in expert_data:
        motion.laterality = expert.laterality

    for motion in student_data:
        motion.laterality = student.laterality

    # Scaling and normalisaing (or not) the data
    # Useful for DBSCAN for example
    scale = param['scale']
    normalise = param['normalise']



    # Algorithm(s) to use
    algos = param['algos']

    clustering_problems = []

    for problem in datatype_joints_list:

        datatype_joints = problem[1]

        expert_sub_data = expert_data[:10] + expert_data[min(expert_data_repartion[problem[0]])-1:max(expert_data_repartion[problem[0]])]

        # if problem[0] == 'javelin':
            # del expert_sub_data[16]
            # del expert_sub_data[15]
            # del expert_sub_data[12]
            # del expert_sub_data[3]
            # del expert_sub_data[2]
            # del expert_sub_data[1]
            # del expert_sub_data[0]

        if problem[0] == 'align_arm':
            del expert_sub_data[15]
            del expert_sub_data[9]
            del expert_sub_data[8]

        for algo, param in algos.items():
            print(problem[0])

            # true_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #                1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            # if problem[0] == 'align_arm':
            #     true_labels = [0, 0, 0, 0, 0, 0, 0, 0,
            #                    1, 1, 1, 1, 1, 1, 1, 1, 1]
            # elif problem[0] == 'leaning':
            #     true_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #                    1, 1, 1, 1, 1, 1, 1, 1, 1]

            model = run_clustering(expert_sub_data, validate_data=False,
                                   datatype_joints=datatype_joints, algorithm=algo,
                                   parameters=param, scale_features=scale,
                                   normalise_features=normalise, true_labels=None,
                                   verbose=False, to_file=True, to_json=True,
                                   return_data=True)

        # Taking the student features
        std_features = data_gathering(student_data, datatype_joints)

        # If the expert data has been scaled, do the same for the student's ones
        if scale:
            scaler = RobustScaler()
            std_features = scaler.fit_transform(std_features)
        # Same for normalising
        if normalise:
            min_max_scaler = MinMaxScaler()
            std_features = min_max_scaler.fit_transform(std_features)

        # Compute the centroid of the student's features (Euclidean distance for now)
        std_centroid = get_centroid(std_features)

        # Compute the distance from the student's centroid to the expert's ones
        distances_to_centroid = compute_distance(model[0].cluster_centers_, std_centroid)

        # Get the distance from the student's centroid to the line between the two expert's centroids
        distance_from_line = get_distance_from_expert_centoids_line(model[0].cluster_centers_, std_centroid)
        # Used to check if the student's centroid is between the expert centroids (diamond shape)
        distance_from_line /= (dst_pts(model[0].cluster_centers_[0], model[0].cluster_centers_[1]) / 2)

        # Normalise the distances to the centroids
        summ = sum(distances_to_centroid)
        for i, distance in enumerate(distances_to_centroid):
            distances_to_centroid[i] = distance/summ

        # Get the most probable cluster label for expert data
        clusters_label = get_cluster_label(expert_sub_data, expert_data_repartion, model[0].labels_)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #TODO: display numbers of motions in labelled clusters#
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        get_closeness_to_clusters(clusters_label, std_features, model[0].cluster_centers_, problem[0], to_csv=False)

        # Display the closeness of the student's data to each expert cluster
        mix_res = mix(distances_to_centroid, clusters_label, distance_from_line)

        clusters_names = get_cluster_labels_from_data_repartition(model[0].labels_, model[0].cluster_centers_)

        # For the rest of the algorithm, if there are more than 2 features,
        # we run the data through a PCA for the next steps
        if len(model[0].cluster_centers_[0]) > 2:

            # model[2] = normalize(model[2])
            # model[0].cluster_centers_ = normalize(model[0].cluster_centers_)
            # std_centroid = normalize(std_centroid.reshape(1, -1))[0]

            pca = PCA(n_components=2, copy=True)
            pca.fit(model[2])

            model[2] = pca.transform(model[2])
            model[0].cluster_centers_ = pca.transform(model[0].cluster_centers_)
            std_centroid = pca.transform(std_centroid.reshape(1, -1))[0]
            std_features = pca.transform(std_features)

        # mixed = np.concatenate((model[2], std_features, model[0].cluster_centers_, std_centroid.reshape(1, -1)), axis=0)
        # min_max_scaler = MinMaxScaler()
        # mixed =  min_max_scaler.fit_transform(mixed)

        # model[2] = mixed[0:len(model[2]), :]
        # std_features = mixed[len(model[2]):len(model[2])+len(std_features), :]
        # model[0].cluster_centers_ = mixed[len(model[2])+len(std_features):-1, :]
        # std_centroid = mixed[-1, :]

        clus_prob = ClusteringProblem(problem=problem[0],
                                      model=model[0], features=model[2],
                                      labels=model[0].labels_,
                                      centroids=model[0].cluster_centers_,
                                      sil_score=model[1]['ss'],
                                      clusters_names=clusters_names,
                                      algo_name=problem[0],
                                      std_data=std_features,
                                      std_centroid=get_centroid(std_features),
                                      distance_to_line=distance_from_line)

        clus_prob.trapezoid = get_trapezoid(model, std_centroid)
        clus_prob.circles = get_circle(model, std_centroid)
        clus_prob.std_data = std_features

        clustering_problems.append(clus_prob)

    give_two_advices(clustering_problems)
    end_msg.hide()
    plot_all_defaults(clustering_problems, only_centroids=True)
    
    if display:
        all_features_to_extract = merge_list_of_dictionnaries([x[1] for x in datatype_joints_list])
        expert_good_data = expert_data[min(expert_data_repartion['good'])-1:max(expert_data_repartion['good'])]
        plot_good_vs_student_all_data(expert_good_data, student_data, algos, all_features_to_extract, only_centroids=True)

    if True:
        expert_bad_data = expert_data[max(expert_data_repartion['good']):]
        good_and_bad_vs_student_all_data(expert_good_data, expert_bad_data, student_data, algos, all_features_to_extract, only_centroids=True)
    
   