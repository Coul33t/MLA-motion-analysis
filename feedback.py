import os
import stat
from shutil import rmtree
from distutils.dir_util import copy_tree

from sklearn.decomposition import PCA

from data_import import json_import

from data_processing import (run_clustering,
                             data_gathering)

from data_visualization import (plot_PCA,
                                multi_plot_PCA,
                                plot_all_defaults)

from feedback_tools import *

from constants import problemes_et_solutions as problems_and_advices

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

def feedback():
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/students/mixed'
    # Expert Data
    name = 'aurel'
    expert_data = import_data(path, name)
    # Student data
    name = 'me'
    student_data = import_data(path, name)

    # Setting the laterality
    for motion in expert_data:
        motion.laterality = 'Right'

    for motion in student_data:
        motion.laterality = 'Left'

    # List of datatypes and joint to process
    # default to check: {Descriptor: [{joint, laterality or not (check left for lefthanded and vice versa)},
    #                                  other joint, laterality or not}],
    #                    Other descriptor: [{joint, laterality or not}]
    #                   }
    datatype_joints_list = []

    datatype_joints_list.append(['leaning', {'MeanSpeed': [{'joint': 'LeftShoulder', 'laterality': False},
                                                           {'joint': 'RightShoulder', 'laterality': False}]
                                }])

    datatype_joints_list.append(['elbow_move', {'MeanSpeed': [{'joint': 'LeftArm', 'laterality': True},
                                                              {'joint': 'LeftShoulder', 'laterality': True}]
                                }])

    datatype_joints_list.append(['javelin', {'PosX': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}],
                                             'PosY': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}],
                                             'PosZ': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}]
                                            }])

    datatype_joints_list.append(['align_arm', {'BoundingBoxMinusX': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusX':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxMinusY': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusY':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxMinusZ': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusZ':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}]
                                              }])



    # Scaling and normalisaing (nor not) the data
    # Useful for DBSCAN for example
    scale = False
    normalise = False

    # Setting the data repartition
    aurelien_data = {'good': [x+1 for x in range(10)],
                     'leaning': [x+1 for x in range(10, 20)],
                     'javelin': [x+1 for x in range(20, 30)],
                     'align_arm': [x+1 for x in range(30, 40)],
                     'elbow_move': [x+1 for x in range(40, 50)]}

    # Algorithm to test
    algos = {'k-means': {'n_clusters': 2}}

    distances_and_clusters = []
    results = []
    std_features_all = []

    for problem in datatype_joints_list:

        datatype_joints = problem[1]
        expert_sub_data = expert_data[:10] + expert_data[min(aurelien_data[problem[0]])-1:max(aurelien_data[problem[0]])]

        for algo, param in algos.items():
            print(problem[0])

            model = run_clustering(path, expert_sub_data, name, validate_data=False,
                                   datatype_joints=datatype_joints, algorithm=algo,
                                   parameters=param, scale_features=scale, normalise_features=normalise,
                                   true_labels=None, verbose=False, to_file=True, to_json=True, return_data=True)

        # Taking the student features
        std_features = data_gathering(student_data, datatype_joints)

        # If the expert data has been scaled, do the same for the student's ones
        if scale:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            std_features = scaler.fit_transform(std_features)
        # Same for normalising
        if normalise:
            from sklearn.preprocessing import MinMaxScaler
            min_max_scaler = MinMaxScaler()
            std_features = min_max_scaler.fit_transform(std_features)

        # Compute the centroid of the student's features (Euclidean distance for now)
        std_centroid = get_centroid_student(std_features)

        # For the rest of the algorithm, if there are more than 2 features,
        # we run the data through a PCA for the next steps
        if len(model[0].cluster_centers_[0]) > 2:
            pca = PCA(n_components=2, copy=True)
            pca.fit(model[2])

            model[2] = pca.transform(model[2])
            model[0].cluster_centers_ = pca.transform(model[0].cluster_centers_)
            std_centroid = pca.transform(std_centroid.reshape(1, -1))[0]
            std_features = pca.transform(std_features)

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
        clusters_label = get_cluster_label(expert_sub_data, aurelien_data, model[0].labels_)
        # Display the closeness of the student's data to each expert cluster
        mix_res = mix(distances_to_centroid, clusters_label, distance_from_line)
        distances_and_clusters.append(mix_res)

        res = (model[0], model[2], model[0].labels_, model[1], problem[0], std_features, get_trapezoid(model, std_centroid), get_circle(model, std_centroid))
        results.append(res)
        std_features_all.append(std_features)

    models = [x[0] for x in results]
    features = [x[1] for x in results]

    for i, feat in enumerate(features):
        features[i] = np.concatenate([feat, std_features_all[i]], axis=0)

    centroids = [x[0].cluster_centers_ for x in results]
    labels = [x[2] for x in results]

    for i, lab in enumerate(labels):
        max_val = max(lab)
        labels[i] = np.concatenate([lab, [max_val+1 for x in range(len(std_features))]], axis=0)

    trapezoids = [x[6] for x in results]
    circles = [x[7] for x in results]

    sss = [x[3]['ss'] for x in results]
    names = ['k-means ' + x[4] for x in results]
    title = 'None'

    clusters_names = [x[0] for x in distances_and_clusters]

    std_data = [[std_feat, [max(labels[i])+1 for x in range(len(std_features_all[i]))]] for i, std_feat in enumerate(std_features_all)]

    multi_plot_PCA(features, labels, clusters_names, names, models, sss, title, trapezoids, circles, only_centroids=True, centroids=centroids, std_data=std_data)

def only_feedback(expert, student, path):
    if path.split('/')[-1] == 'mixed':
        take_last_data("/".join(path.split('/')[:-1]), student, expert, number=9)
    else:
        take_last_data(path, student, expert, number=9)

    # Expert Data
    expert_data = import_data(path, expert.name)
    # Student data
    student_data = import_data(path, student.name)

    # Setting the laterality
    for motion in expert_data:
        motion.laterality = expert.laterality

    for motion in student_data:
        motion.laterality = student.laterality

    # List of datatypes and joint to process
    # default to check: {Descriptor: [{joint, laterality or not (check left for lefthanded and vice versa)},
    #                                  other joint, laterality or not}],
    #                    Other descriptor: [{joint, laterality or not}]
    #                   }
    datatype_joints_list = []

    datatype_joints_list.append(['leaning', {'MeanSpeed': [{'joint': 'LeftShoulder', 'laterality': False},
                                                           {'joint': 'RightShoulder', 'laterality': False}]
                                }])

    datatype_joints_list.append(['elbow_move', {'MeanSpeed': [{'joint': 'LeftArm', 'laterality': True},
                                                              {'joint': 'LeftShoulder', 'laterality': True}]
                                }])

    datatype_joints_list.append(['javelin', {'PosX': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}],
                                             'PosY': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}],
                                             'PosZ': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}]
                                            }])

    datatype_joints_list.append(['align_arm', {'BoundingBoxMinusX': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusX':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxMinusY': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusY':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxMinusZ': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusZ':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}]
                                              }])



    # Scaling and normalisaing (nor not) the data
    # Useful for DBSCAN for example
    scale = False
    normalise = False

    aurelien_data = {'good': [x+1 for x in range(10)],
                     'leaning': [x+1 for x in range(10, 19)],
                     'javelin': [x+1 for x in range(20, 30)],
                     'align_arm': [x+1 for x in range(30, 40)],
                     'elbow_move': [x+1 for x in range(40, 50)]}

    # Algorithm to test
    algos = {'k-means': {'n_clusters': 2}}

    clustering_problems = []

    for problem in datatype_joints_list:

        datatype_joints = problem[1]

        expert_sub_data = expert_data[:10] + expert_data[min(aurelien_data[problem[0]])-1:max(aurelien_data[problem[0]])]

        for algo, param in algos.items():
            print(problem[0])

            model = run_clustering(expert_sub_data, validate_data=False,
                                   datatype_joints=datatype_joints, algorithm=algo,
                                   parameters=param, scale_features=scale, normalise_features=normalise,
                                   true_labels=None, verbose=False, to_file=True, to_json=True, return_data=True)

        # Taking the student features
        std_features = data_gathering(student_data, datatype_joints)

        # If the expert data has been scaled, do the same for the student's ones
        if scale:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            std_features = scaler.fit_transform(std_features)
        # Same for normalising
        if normalise:
            from sklearn.preprocessing import MinMaxScaler
            min_max_scaler = MinMaxScaler()
            std_features = min_max_scaler.fit_transform(std_features)

        # Compute the centroid of the student's features (Euclidean distance for now)
        std_centroid = get_centroid_student(std_features)

        # For the rest of the algorithm, if there are more than 2 features,
        # we run the data through a PCA for the next steps
        if len(model[0].cluster_centers_[0]) > 2:
            pca = PCA(n_components=2, copy=True)
            pca.fit(model[2])

            model[2] = pca.transform(model[2])
            model[0].cluster_centers_ = pca.transform(model[0].cluster_centers_)
            std_centroid = pca.transform(std_centroid.reshape(1, -1))[0]
            std_features = pca.transform(std_features)

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
        clusters_label = get_cluster_label(expert_sub_data, aurelien_data, model[0].labels_)
        # Display the closeness of the student's data to each expert cluster
        mix_res = mix(distances_to_centroid, clusters_label, distance_from_line)

        clusters_names = get_cluster_labels_from_data_repartition(model[0].labels_, model[0].cluster_centers_)

        clus_prob = ClusteringProblem(problem=problem[0],
                                      model=model[0], features=model[2],
                                      labels=model[0].labels_,
                                      centroids=model[0].cluster_centers_,
                                      sil_score=model[1]['ss'],
                                      clusters_names=clusters_names,
                                      algo_name=problem[0],
                                      std_data=std_features,
                                      std_centroid=get_centroid_student(std_features),
                                      distance_to_line=distance_from_line)

        clus_prob.trapezoid = get_trapezoid(model, std_centroid)
        clus_prob.circles = get_circle(model, std_centroid)
        clus_prob.std_data = std_features

        clustering_problems.append(clus_prob)

    give_two_advices(clustering_problems)
    plot_all_defaults(clustering_problems, only_centroids=True)

def take_last_data(path, student, expert, number=9):

    if not student.full_name in os.listdir(path):
        print(f'ERROR: {student.full_name} not found in {path}.')
        return False

    mixed_path = os.path.normpath(os.path.join(path, 'mixed'))

    # Remove all non-expert data )
    for file in os.listdir(mixed_path):
        if expert.name not in file:
            rmtree(os.path.normpath(os.path.join(mixed_path, file)))

    std_path = os.path.normpath(os.path.join(path, student.full_name))

    file_list = os.listdir(std_path)
    file_list = sorted(file_list, key=lambda sin: int(sin.replace(student.name + '_', '').replace('Char00', '')))
    folders_to_copy = file_list[-number:]

    for folder_to_copy in folders_to_copy:
        std_folder = os.path.join(std_path, folder_to_copy)
        print(f'doing {std_folder}')
        copy_tree(std_folder, os.path.join(mixed_path, folder_to_copy))

if __name__ == '__main__':
    expert = Person(r'', 'aurel', 'Right')
    student = Person(r'', '', '', '')
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/students/mixed'
    only_feedback(expert, student, path)