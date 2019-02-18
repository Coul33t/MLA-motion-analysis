import constants as cst
import data_labels as dl

from data_import import json_import

from data_processing import (test_full_batch_k_var,
                             k_mean_mixed,
                             k_means_second_pass,
                             k_means_second_pass_all_data,
                             run_clustering)

from data_visualization import (plot_PCA,
                                multi_plot_PCA)

import pdb

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

def select_one_class(original_data, labels, c_to_keep):
    new_original_data = []

    for i, motion in enumerate(original_data):
        if labels[i] == c_to_keep:
            new_original_data.append(motion)

    return new_original_data


def set_dominant(list_of_motions):
    for motion in list_of_motions:
        if 'Aous' in motion.name or 'Damien' in motion.name:
            motion.laterality = 'Left'
        else:
            motion.laterality = 'Right'

def main_all_joints():

    people_names = cst.people_names_O
    data_types_combination = cst.data_types_combination
    right_joints_list = cst.right_joints_list
    left_joints_list = cst.left_joints_list

    people_names = [[['Esteban', 'Guillaume', 'Ines', 'Iza', 'Ludovic', 'Marc', 'Oussema', 'Pierre', 'Sebastien', 'Vincent', 'Yann'], 'right']]

    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/LALALTESTNEWMAX/'

    for people in people_names:

        name = people[0]
        joint_list = right_joints_list

        if people[1] == 'left':
            joint_list = left_joints_list

        original_data = import_data(path, [name])

        for data_to_select in data_types_combination:
            print(f'\n\n\nProcessing {name}...')
            test_full_batch_k_var(path,
                                  original_data,
                                  [name],
                                  validate_data=False,
                                  joint_to_use=joint_list,
                                  data_to_select=data_to_select,
                                  true_labels=None,
                                  verbose=False,
                                  to_file=True,
                                  to_json=True,
                                  display_graph=False,
                                  save_graph=True,
                                  data_to_graph='ss',
                                  only_success=False)

def main_all_together():
    people_names = cst.people_names
    data_types_combination = cst.data_types_combination
    joint_list = cst.neutral_joints_list
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/Bottle_Flip_Challenge/glu/'

    original_data = []
    labels = []

    for people in people_names:

        name = people[0]
        if name != 'Aous' and name != 'Damien':
            print(f'Importing {name} motion data')
            original_data.extend(import_data(path, [name]))
            labels.extend(getattr(dl, cst.names_labels[name]))

    print(f'Setting laterality')
    set_dominant(original_data)

    results = []
    for data_to_select in data_types_combination:
        print(f'\n\n\nProcessing all for {data_to_select}')
        results.append(k_mean_mixed(path, original_data, ['all'], validate_data=False,
                                    joint_to_use=joint_list, data_to_select=data_to_select,
                                    true_labels=labels, verbose=False, to_file=True,
                                    to_json=True, display_graph=False, save_graph=True,
                                    data_to_graph='ss'))


def main_leo():
    data_types_combination = cst.data_types_combination

    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/Throw_ball/'

    name = 'Leo'
    joint_list =  cst.right_joints_list

    original_data = import_data(path, [name])
    # original_data = select_one_class(original_data, dl.LEO_THROW_LABELS, 1)
    # labels = dl.LEO_LABELS_2[50:]
    labels =dl.LEO_THROW_TYPES

    for data_to_select in data_types_combination:
        print(f'\n\n\nProcessing {name}...')
        test_full_batch_k_var(path,
                              original_data,
                              [name],
                              validate_data=False,
                              joint_to_use=joint_list,
                              data_to_select=data_to_select,
                              true_labels=labels,
                              verbose=False,
                              to_file=True,
                              to_json=True,
                              display_graph=False,
                              save_graph=True,
                              data_to_graph='ars',
                              only_success=False)

def main_second_pass():
    for folder in return_files(r'C:/Users/quentin/Documents/These/Databases/Res/all_ss_05/'):
        for json_file in return_files(r'C:/Users/quentin/Documents/These/Databases/Res/all_ss_05/' + folder, 'json'):
            print(f'Processing {folder}: {json_file}')
            k_means_second_pass(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed',
                                r'C:/Users/quentin/Documents/These/Databases/Res/all_ss_05/' + folder + '/',
                                json_file,
                                folder)

def main_second_pass_all_data():
    for folder in return_files(r'C:/Users/quentin/Documents/These/Databases/Res/all_ss_05/'):
        for json_file in return_files(r'C:/Users/quentin/Documents/These/Databases/Res/all_ss_05/' + folder, 'json'):
            if folder == 'Right-Handed' or folder == 'Sebastien' or folder == 'Vincent' or folder == 'Yann':
                print(f'Processing {folder}: {json_file}')
                k_means_second_pass_all_data(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed',
                                             r'C:/Users/quentin/Documents/These/Databases/Res/all_ss_05/' + folder + '/',
                                             json_file,
                                             folder)

def test():
    people_names = cst.people_names
    data_types_combination = cst.data_types_combination
    joint_list = cst.right_joints_list
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/Bottle_Flip_Challenge/glu/'

    original_data = []
    labels = []

    for people in people_names:

        name = people[0]

        print(f'Importing {name} motion data')
        original_data.extend(import_data(path, [name]))
        labels.extend(getattr(dl, cst.names_labels[name]))


    for i,people in enumerate(people_names):
        tst = import_data(path, [people[0]])
        valid = True
        for j in range(100):
            if tst[i].get_datatype('BegMaxEndSpeedNorm').get_joint('LeftHand') != original_data[(i*100) + i].get_datatype('BegMaxEndSpeedNorm').get_joint('LeftHand'):
                valid = False

        print(valid)

def darts_test():
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/testBB/'
    name = 'me'
    original_data = import_data(path, [name])
    joints = ['LeftArmLeftForeArmLeftHandLeftShoulder']
    data = ['BoundingBoxMinusX', 'BoundingBoxPlusX',
          'BoundingBoxMinusY', 'BoundingBoxPlusY',
          'BoundingBoxMinusZ', 'BoundingBoxPlusZ']

    original_data = original_data[:30]

    algos = {'k-means': {'n_clusters': 2},
             'dbscan': {'eps': 12, 'min_samples': 2},
             'agglomerative': {'n_clusters': 2},
             'mean-shift': {},
             'gmm': {'n_components': 2}}


    names = []
    labels = []
    features = []

    for algo, param in algos.items():
        res = run_clustering(path, original_data, name, validate_data=False,
                             joint_to_use=joints, data_to_select=data, algorithm=algo,
                             parameters=param, true_labels=None, verbose=False, to_file=True,
                             to_json=True, display_graph=False, save_graph=False,
                             data_to_graph=None, only_success=False, return_data=True)

        print(f'{algo} with {param}: {res[0].labels_}')
        if 'ss' in res[1]:
            print(f'Silhouette Score: {res[1]["ss"]}')
            names.append(algo)
            labels.append(res[0].labels_)
            features.append(res[2])

    features.append(features[0])
    labels.append(dl.FAKE_DARTS_LABELS[:30])
    names.append('Ground Truth')
    multi_plot_PCA(features, labels, names)

def find_optimal_dbscan_eps():
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/testBB/'
    name = 'me'
    original_data = import_data(path, [name])
    joints = ['LeftArmLeftForeArmLeftHandLeftShoulder']
    data = ['BoundingBoxMinusX', 'BoundingBoxPlusX',
          'BoundingBoxMinusY', 'BoundingBoxPlusY',
          'BoundingBoxMinusZ', 'BoundingBoxPlusZ']

    original_data = original_data[0:40]

    eps = 0
    ss_s = 0
    nb_min= 0
    labels = []

    features = []

    for i in range(2,34):
        for j in range(1,len(original_data)-1):
            res = run_clustering(path, original_data, name, validate_data=False,
                                 joint_to_use=joints, data_to_select=data, algorithm='dbscan',
                                 parameters={'eps': i/2, 'min_samples': j}, true_labels=None, verbose=False, to_file=True,
                                 to_json=True, display_graph=False, save_graph=False,
                                 data_to_graph=None, only_success=False, return_data=True)

            if 'ss' in res[1]:
                ssres = res[1]['ss']
                if ssres > ss_s:
                    ss_s = ssres
                    eps = i/2
                    nb_min = j
                    labels = res[0].labels_
                    features = res[2]

    print(f'dbscan with eps={eps} and min_samples={nb_min}: {labels}')
    print(f'Silhouette Score: {ss_s}')

    plot_PCA(features, labels)

if __name__ == '__main__':
    # test()
    # main_all_together()
    # main_all_joints()
    # main_leo()
    darts_test()
    # find_optimal_dbscan_eps()
    # original_data = json_import(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/', 'TEST_VIS')

    # data_to_select = [['Norm'], ['NewThrowNorm']]

    # # We keep the data we're interested in (from the .json file)
    # selected_data = joint_selection(original_data, 'RightHand')

    # # We put them in the right shape for the algorithm [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
    # features1 = data_selection(selected_data, data_to_select[0])
    # features2 = data_selection(selected_data, data_to_select[1])
    # simple_plot_2d_2_curves(features1, features2)
    # plot_speed(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/LALALASEB/', 'Sebastien_37Char00', 'RightHand')

    # main_second_pass()

    # main_second_pass_all_data()
