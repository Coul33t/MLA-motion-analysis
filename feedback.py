from math import floor, sqrt

import operator

import numpy as np
from scipy.spatial import distance

import matplotlib.path as mplPath

from dataclasses import dataclass

from sklearn.decomposition import PCA

from data_import import json_import

from data_processing import (run_clustering,
                             data_gathering)

from data_visualization import (plot_PCA,
                                multi_plot_PCA,
                                plot_all_defaults)

from constants import problemes_et_solutions as problems_and_advices
@dataclass
class Circle:
    def __init__(self, center, radius, limits=None, is_good=False):
        self.center = center
        self.radius = radius
        self.limits = limits
        # TODO: add limits check
        self.is_good = is_good

@dataclass
class Trapezoid:
    def __init__(self, p1, p2, p3, p4, path=None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.path = path
        if not self.path:
            self.path = mplPath.Path(np.array([p1, p2, p3, p4]))

    def get_max_distance(self):
        return max([np.linalg.norm(x - y) for x in [np.asarray(self.p1), np.asarray(self.p2), np.asarray(self.p3), np.asarray(self.p4)]
                                          for y in [np.asarray(self.p1), np.asarray(self.p2), np.asarray(self.p3), np.asarray(self.p4)]])

@dataclass
class ClusteringProblem:
    def __init__(self, problem=None, model=None, features=None,
                 centroids=None, labels=None, sil_score=None,
                 algo_name=None, clusters_names=None, std_data=None,
                 std_centroid=None, trapezoid=None, circles=None,
                 distance_to_line=None):
        self.problem = problem
        self.model = model
        self.features = features
        self.centroids = centroids
        self.labels = labels
        self.sil_score = sil_score
        self.algo_name = algo_name
        self.clusters_names = clusters_names

        self.std_data = std_data
        self.std_centroid = std_centroid

        self.trapezoid = trapezoid
        self.circles = circles

        self.distance_to_line = distance_to_line

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

def intra_cluster_dst(centroid, points):
    return np.mean(np.asarray([np.linalg.norm(point - centroid) for point in points]))

def compute_distance(centroids, feature):
    return [np.linalg.norm(feature - centroid) for centroid in centroids]

def dst_pts(pt1, pt2):
    return sqrt(pow(pt2[0] - pt1[0], 2) + pow(pt2[1] - pt1[1], 2))

def get_trapezoid(result, point):
    centroids = result[0].cluster_centers_
    points = [result[2][np.where(result[0].labels_ == i)] for i in np.unique(result[0].labels_)]
    max_dst = [max(max(distance.cdist([centroids[i]], cluster, 'euclidean'))) for i, cluster in enumerate(points)]

    p1 = (centroids[0][0], centroids[0][1] + max_dst[0])
    p4 = (centroids[0][0], centroids[0][1] - max_dst[0])
    p2 = (centroids[1][0], centroids[1][1] + max_dst[1])
    p3 = (centroids[1][0], centroids[1][1] - max_dst[1])

    trapezoid = Trapezoid(p1, p2, p3, p4)
    # trapezoid = mplPath.Path(np.array([p1, p2, p3, p4]))
    # print(f"Point is in trapezoid: {is_in_trapezoid(point, trapezoid.path)}")
    return trapezoid

def is_in_trapezoid(point, trapezoid):
    return trapezoid.contains_point(point)

def get_circle(result, point):
    centroids = result[0].cluster_centers_
    points = [result[2][np.where(result[0].labels_ == i)] for i in np.unique(result[0].labels_)]

    good_or_bad = np.ndarray.tolist(np.where(result[0].labels_ == 0)[0])

    is_good = False
    if len(set(good_or_bad).intersection([x for x in range(10)])) > 5:
        is_good = True

    max_dst = [max(max(distance.cdist([centroids[i]], cluster, 'euclidean'))) for i, cluster in enumerate(points)]

    med_dst = [intra_cluster_dst(centroid, points[i]) for i, centroid in enumerate(centroids)]

    is_in_cluster = is_in_circle(point, centroids[0], max_dst[0])
    if not is_in_cluster:
        is_in_cluster = is_in_circle(point, centroids[1], max_dst[1])

    circle_good = Circle(centroids[0], max_dst[0], limits={'center': centroids[0], 'radius_max':max_dst[0], 'radius_med':(max_dst[0] + med_dst[0]) / 2, 'radius_min':med_dst[0]}, is_good=is_good)
    circle_bad = Circle(centroids[1], max_dst[1], limits={'center': centroids[1], 'radius_max':max_dst[1], 'radius_med':(max_dst[1] + med_dst[1]) / 2, 'radius_min':med_dst[1]}, is_good=not is_good)
    # print(f"Point is in a cluster: {is_in_cluster}")
    return (circle_good, circle_bad)

def is_in_circle(point, centroid, diameter):
    return np.linalg.norm(point - centroid) < diameter

def is_in_circle_c(point, circle):
    return np.linalg.norm(point - circle.center) < circle.radius

def get_cluster_label(original_data, original_labels, clustering_labels):

    labelled_cluster = {}
    labelled_data = []
    for data in original_data:
        # Get rid of the name + the Char00

        # number = int(data.name.split('_')[1][:-6])
        number = int(data.name.split("_")[-2])

        for k, v in original_labels.items():
            if number in v:
                number = k
                break

        if not isinstance(number, int):
            labelled_data.append(number)

    c_rep = []
    for cluster in set(clustering_labels):
        c_rep.append({x: 0 for x in original_labels.keys()})
        for i, elem in enumerate(clustering_labels):
            if elem == cluster:
                c_rep[-1][labelled_data[i]] += 1


    for i, cluster in enumerate(c_rep):
        labelled_cluster[f'c{i}'] = max(cluster.items(), key=operator.itemgetter(1))[0]

    return labelled_cluster

def get_centroid_student(std_data):
    return np.mean(std_data, axis=0)

def get_distance_from_expert_centoids_line(exp_centroids, std_centroid):
    pt1 = {'x': exp_centroids[0][0], 'y': exp_centroids[0][1]}
    pt2 = {'x': exp_centroids[1][0], 'y': exp_centroids[1][1]}

    pt_std = {'x': std_centroid[0], 'y': std_centroid[1]}

    A = pt2['y'] - pt1['y']
    B = pt2['x'] - pt1['x']
    C = (pt1['x'] * pt2['y']) - (pt2['x'] * pt1['y'])

    return abs((A*pt_std['x']) + (B*pt_std['y']) + C) / sqrt(pow(A, 2) + pow(B, 2))

def mix(distances, c_labels, distance_line_vs_distance_centroids):
    new_dict = {}
    for i, (_, v) in enumerate(c_labels.items()):
        new_dict[v] = distances[i]


    keys_to_display = list(new_dict.keys())
    if keys_to_display[0] != 'good':
        keys_to_display.reverse()

    size_display = 40
    char = '#'
    place = floor((size_display/100) * (100 * new_dict['good']))

    print(f"{keys_to_display[0]} [", end='')
    for i in range(size_display):
        if i == place:
            print(char, end='')
        else:
            print('-', end='')
    print(f"] {keys_to_display[1]} ({distance_line_vs_distance_centroids:.2f})")

    return (keys_to_display, distance_line_vs_distance_centroids)


def only_feedback():
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/students/mixed'
    # Expert Data
    name = 'aurel'
    expert_data = import_data(path, name)
    # Student data
    name = 'CorgniardA'
    student_data = import_data(path, name)


    # Setting the laterality
    for motion in expert_data:
        motion.laterality = 'Right'

    for motion in student_data:
        motion.laterality = 'Right'

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

    expert_data.pop(19)

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

def get_cluster_labels_from_data_repartition(labels, centroids):
    good_or_bad = np.ndarray.tolist(np.where(labels == 0)[0])

    if len(set(good_or_bad).intersection([x for x in range(10)])) > 5:
        return('good', 'bad')

    return ('bad', 'good')

def give_advice(clustering_problems):
    candidates = [x for x in clustering_problems if is_in_trapezoid(x.std_centroid, x.trapezoid.path)]
    if candidates:
        # dst(std, good) / dst(good, bad) -> normalisation
        dst_to_closest = [np.linalg.norm((x.std_centroid - x.centroids[x.clusters_names.index('good')]) / x.trapezoid.get_max_distance()) for x in candidates]
        print(dst_to_closest)
        advice_to_give = candidates[dst_to_closest.index(min(dst_to_closest))].problem
        print(f"Problem: {advice_to_give}")
        print(problems_and_advices[advice_to_give])

def get_indexes(candidates):
    first_dst = None
    second_dst = None

    all_dst = []

    if len(candidates) > 0:
        all_dst = [np.linalg.norm((x.std_centroid - x.centroids[x.clusters_names.index('good')]) / x.trapezoid.get_max_distance()) for x in candidates]
        first_dst = all_dst.index(min(all_dst))

    if len(candidates) > 1:
        all_dst[first_dst] = max(all_dst) + 1
        second_dst = all_dst.index(min(all_dst))

    return first_dst, second_dst

def give_two_advices(clustering_problems):
    candidates = [x for x in clustering_problems if (is_in_trapezoid(x.std_centroid, x.trapezoid.path)
                                                 or is_in_circle_c(x.std_centroid, next((y for y in x.circles if y.is_good == False), None)))
                                                     and not is_in_circle_c(x.std_centroid, next((y for y in x.circles if y.is_good == True), None))]

    if len(candidates) > 1:
        print(f"2 or more candidates ({len(candidates)})")

        # dst(std, good) / dst(good, bad) -> normalisation
        first_advice_idx, second_advice_idx = get_indexes(candidates)

        advice_to_give = candidates[first_advice_idx].problem
        print(f"Problem: {advice_to_give}\n {problems_and_advices[advice_to_give]}")
        advice_to_give = candidates[second_advice_idx].problem
        print(f"Problem: {advice_to_give}\n {problems_and_advices[advice_to_give]}")

    elif len(candidates) == 1:
        print("1 candidate")

        # dst(std, good) / dst(good, bad) -> normalisation
        first_dst, _ = get_indexes(candidates)
        print(f"Problem: {advice_to_give}\n {problems_and_advices[advice_to_give]}")

        dst_to_line = [x.distance_to_line for x in clustering_problems]
        second_advice = dst_to_line.index(min(dst_to_line))
        dst_to_line[second_advice] = max(dst_to_line) + 1
        second_advice = dst_to_line.index(min(dst_to_line))
        advice_to_give = clustering_problems[second_advice].problem
        print(f"Problem: {advice_to_give}")
        print(problems_and_advices[advice_to_give])

    else:
        print("No candidate")

        max_val = max([x.distance_to_line for x in clustering_problems])
        dst_to_line = []
        for i, x in enumerate(clustering_problems):
            if (not is_in_trapezoid(x.std_centroid, x.trapezoid.path)
                and not is_in_circle_c(x.std_centroid, next((y for y in x.circles if y.is_good == True), None))):
                dst_to_line.append(x.distance_to_line)
            else:
                dst_to_line.append(max_val + 1)


        first_advice = dst_to_line.index(min(dst_to_line))
        advice_to_give = clustering_problems[first_advice].problem
        print(f"Problem: {advice_to_give}\n {problems_and_advices[advice_to_give]}")

        dst_to_line[first_advice] = max(dst_to_line) + 1
        second_advice = dst_to_line.index(min(dst_to_line))
        advice_to_give = clustering_problems[second_advice].problem
        print(f"Problem: {advice_to_give}\n {problems_and_advices[advice_to_give]}")

if __name__ == '__main__':
    only_feedback()