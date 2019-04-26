# Internal libraries
import os

from math import ceil

from shutil import rmtree
from distutils.dir_util import copy_tree

# external libraries

from sklearn.decomposition import PCA

# Personnal code
from data_import import json_import

from data_processing import (run_clustering,
                             data_gathering)

from data_visualization import plot_progression

from feedback_tools import *

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

def copy_data(expert_path, student_path, output_path, student_to_copy=None):
    if os.path.exists(output_path):
        print(f"WARNING: path\n{output_path}\nalready exists! Do you want to continue (the folder and its contents WILL BE deleted)? (y/n)")
        res = input()
        if not res.lower() == 'y':
            print("ABORTED.")
            return False
        rmtree(output_path)

    os.makedirs(output_path)

    copy_tree(expert_path, output_path)
    copy_tree(student_path, output_path)

    return True


def delete_path(path):
    rmtree(path)


def progression(expert, student, tmp_path):
    if not copy_data(expert.path, student.path, tmp_path):
        return

    # Expert Data
    expert_data = import_data(tmp_path, expert.name)
    # Student data
    student_data = import_data(tmp_path, student.name)


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

    clustering_problems = [[] for x in range(4)]
    student_sep = [(i*9, (i*9)+9) for i in range(4)]

    # For the four 9 successives throws
    for current_throws in range(4):

        # For each possible problem
        for j, problem in enumerate(datatype_joints_list):
            print(f"Doing {problem[0]} problem for the {student_sep[current_throws]} throws")

            datatype_joints = problem[1]

            expert_sub_data = expert_data[:10] + expert_data[min(aurelien_data[problem[0]])-1:max(aurelien_data[problem[0]])]

            for algo, param in algos.items():
                print(problem[0])

                model = run_clustering(expert_sub_data, validate_data=False,
                                       datatype_joints=datatype_joints, algorithm=algo,
                                       parameters=param, scale_features=scale, normalise_features=normalise,
                                       true_labels=None, verbose=False, to_file=True, to_json=True, return_data=True)

            # Taking the student features
            std_features = data_gathering(student_data[student_sep[current_throws][0]:student_sep[current_throws][1]], datatype_joints)

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

            clustering_problems[j].append(clus_prob)

    delete_path(tmp_path)

    title_to_print = "Progression de l'apprenant (" + student.path.split('/')[-1].replace('_', ' ') + ")"
    plot_progression(clustering_problems, title=title_to_print, text=None)

if __name__ == "__main__":
    expert = Person(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/aurelien', 'aurel', 'Right')
    student = Person(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/students/Sicard_Teddy', 'SicardT', 'Right')
    tmp_path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/tmp_path'
    progression(expert, student, tmp_path)
    # if copy_data(expert.path, student.path, tmp_path, student_to_copy=[x for x in range(9)]):
    #     delete_path(tmp_path)