# https://artem.krylysov.com/blog/2015/09/29/benchmark-python-json-libraries/
try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json

import pdb
import os

from collections import OrderedDict

from tools import file_name_gathering, natural_keys, merge_dict
from motion_classes.motion import Motion


def return_files(f, str_to_check=[''], present=True):
    """
        returns a list of files in a folder
        if str_to_check is or isn't in the names
        (default: all files)
    """
    if present:
        if isinstance(str_to_check, list):
            return sorted([x for x in os.listdir(f) if all(sub_str in x for sub_str in str_to_check)], key=natural_keys)
        elif isinstance(str_to_check, str):
            return sorted([x for x in os.listdir(f) if str_to_check in x], key=natural_keys)
    else:
        if isinstance(str_to_check, list):
            return sorted([x for x in os.listdir(f) if all(sub_str not in x for sub_str in str_to_check)], key=natural_keys)
        elif isinstance(str_to_check, str):
            return sorted([x for x in os.listdir(f) if str_to_check not in x], key=natural_keys)


def return_data(f, joints_to_append=None):
    """
        returns a list, containing:
        [list motion[list seg[list speed[OrderedDict joints(name, value)]]]]
    """
    full_data = []

    data = OrderedDict()

    for folder in return_files(f):

        motion_data = []

        for subfolders in return_files(f + '/' + folder, '.json', False):

            seg_data = []

            for file in return_files (f + '/' + folder + '/' + subfolders, '.csv', True):

                seg_data.append(file_gathering_dict(f + '/' + folder + '/' + subfolders + '/' + file))
                print(folder + ' ' + subfolders + ' ' + file)

            motion_data.append(seg_data)

        full_data.append(motion_data)


    return full_data

def return_data_with_names(f, joints_to_append=None):
    """
        returns a list, containing:
        full_data: [motion name, motion data]
        motion data: []
    """
    files_list = []
    full_data = []

    data = OrderedDict()

    for folder in return_files(f):

        motion_data = []

        for subfolder in return_files(f + '/' + folder, '.json', False):

            seg_data = []

            for file in return_files (f + '/' + folder + '/' + subfolder, '.csv', True):

                seg_data.append([file, file_gathering_dict(f + '/' + folder + '/' + subfolder + '/' + file)])
                print(folder + ' ' + subfolder + ' ' + file)

            motion_data.append([subfolder, seg_data])

        full_data.append([folder, motion_data])

    return full_data

def json_import(folder_path, folder_name=None):
    """
        Import the data from json files. Each motion is put into a Motion class.
    """
    if not folder_name:
        folder_name = ['JSON_BATCH_TEST']

    full_data = []

    for folder in return_files(folder_path, folder_name):
        for subfolders in return_files(folder_path + '/' + folder, '.json', False):

            file_list = file_name_gathering(folder_path + '/' + folder + '/' + subfolders)
            file_list.sort(key=natural_keys)


            for file in file_list:

                # Stripping the '.json'
                motion = Motion(name=file[:-5])

                try:
                    motion.pre_processing_info = json.load(open(folder_path + '/' + folder + '/' + 'motion_information.json'))
                except FileNotFoundError:
                    # print(f'WARNING: no pre-processing information for {folder}')
                    pass

                try:
                    motion.post_processing_info = json.load(open(folder_path + '/' + folder + '/' + 'segmentation_information.json'))
                except FileNotFoundError:
                    # print(f'WARNING: no post-processing information for {folder}')
                    pass

                with open(folder_path + '/' + folder + '/' + subfolders + '/' + file, 'r') as f:
                    json_file = json.load(f)
                    for key in list(json_file.keys()):
                        motion.add_datatype(key, json_file[key])

                full_data.append(motion)

    return full_data

def json_specific_import(folder_path, file_list):
    """
        Import the data from specified json files. Each motion is put into a Motion class.
    """
    full_data = []

    if isinstance(file_list, str):
        file_list = [file_list]

    for file in file_list:
        motion = Motion(name=file)
        try:
            motion.pre_processing_info = json.load(open(folder_path + '/' + file + '/' + 'motion_information.json'))
        except FileNotFoundError:
            pass

        try:
            motion.post_processing_info = json.load(open(folder_path + '/' + file + '/' + 'segmentation_information.json'))
        except FileNotFoundError:
            pass

        with open(folder_path + '/' + file + '/data/' + file + '.json', 'r') as f:
            json_file = json.load(f)
            for key in list(json_file.keys()):
                motion.add_datatype(key, json_file[key])

        full_data.append(motion)

    return full_data


if __name__ == '__main__':
    breakpoint()
