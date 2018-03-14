import json

import pdb
import os

from collections import OrderedDict

from tools import file_name_gathering, natural_keys, merge_dict
from motion_classes.motion import *


def data_gathering_dict(folder_path, joints_to_append=None):
    """
        Put the data from multiples files (in the folder folder_path) into a dictionnary.
        Each key is associated with a set of values.
        joints_to_append is there if you want only certains joints.
        Ex : {"Head":[0.25,0,1.25], "Neck":[0.15,0.10,0.75]}
    """
    file_list = file_name_gathering(folder_path)

    # Oh my god I'm so ashamed it took me so long to find this ...
    # Sort the files into natural order (instead of string order)
    file_list.sort(key=natural_keys)

    data = OrderedDict()

    # For each file in the folder
    for filename in file_list:

        # Open the file
        with open(folder_path+'\\'+filename, 'r') as file_data:
            text = file_data.read()

            # For each line
            for line in enumerate(text.split('\n')):
                splitted_line = line[1].split(',')

                # if there's actually something (IF not_empty AND first_value AND second_value)
                if splitted_line and splitted_line[0] and splitted_line[1]:

                    # If there's no joints_to_append parameter, then do it all time
                    # If there's a joints_to_append AND the name of the joint is in joints_to_append, do it
                    if (not joints_to_append) or (joints_to_append and splitted_line[0] in joints_to_append):
                        # setdefault() append to they value if the key exists, else it adds the key and append
                        data.setdefault(splitted_line[0], []).append(float(splitted_line[1]))
                    
    return data


def data_gathering_full(folder_path, joints_to_append=None):

    full_data = []

    for folder in return_files(folder_path):
        
        motion_data = []
        
        for subfolders in return_files(folder_path + '\\' + folder, '.json', False):
            print(folder_path + '\\' + folder + '\\' + subfolders)
            full_data.append(data_gathering_dict(folder_path + '\\' + folder + '\\' + subfolders))

    return full_data
    

def adhoc_gathering(folder_path, joints_to_append=None):
    full_data = []

    for folder in return_files(folder_path):
        
        motion_data = []
        tmp_data = []

        for subfolders in return_files(folder_path + '\\' + folder, '.json', False):
            tmp_data.append(data_gathering_dict(folder_path + '\\' + folder + '\\' + subfolders, joints_to_append))

        tmp_data = merge_dict(tmp_data)
        full_data.append(tmp_data)

    return full_data

def file_gathering_dict(file_path, joints_to_append=None):
    """
        returns an ordered dict (joint name, value)
        from a .csv file
    """
    data = OrderedDict()

    # Open the file
    with open(file_path, 'r') as file_data:
        text = file_data.read()

        # For each line
        for line in enumerate(text.split('\n')):
            splitted_line = line[1].split(',')

            # if there's actually something (IF not_empty AND first_value AND second_value)
            if splitted_line and splitted_line[0] and splitted_line[1]:

                # If there's no joints_to_append parameter, then do it all time
                # If there's a joints_to_append AND the name of the joint is in joints_to_append, do it
                if (not joints_to_append) or (joints_to_append and splitted_line[0] in joints_to_append):
                    # setdefault() append to they value if the key exists, else it adds the key and append
                    data.setdefault(splitted_line[0], []).append(float(splitted_line[1]))
                    

    return data


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
        
        for subfolders in return_files(f + '\\' + folder, '.json', False):
            
            seg_data = []
            
            for file in return_files (f + '\\' + folder + '\\' + subfolders, '.csv', True):

                seg_data.append(file_gathering_dict(f + '\\' + folder + '\\' + subfolders + '\\' + file))
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
        
        for subfolder in return_files(f + '\\' + folder, '.json', False):
            
            seg_data = []
            
            for file in return_files (f + '\\' + folder + '\\' + subfolder, '.csv', True):

                seg_data.append([file, file_gathering_dict(f + '\\' + folder + '\\' + subfolder + '\\' + file)])
                print(folder + ' ' + subfolder + ' ' + file)

            motion_data.append([subfolder, seg_data])

        full_data.append([folder, motion_data])
               
    return full_data

def json_import(folder_path, folder_name=None):
    """
        Import the data from json files. Each motion is put into a Motion class
    """
    if not folder_name:
        folder_name = ['JSON_BATCH_TEST']

    full_data = []

    for folder in return_files(folder_path, folder_name):
        for subfolders in return_files(folder_path + '\\' + folder, '.json', False):

            file_list = file_name_gathering(folder_path + '\\' + folder + '\\' + subfolders)
            file_list.sort(key=natural_keys)


            for file in file_list:
                
                # Stripping the '.json'
                motion = Motion(name=file[:-5])
                motion.pre_processing_info = json.load(open(folder_path + '\\' + folder + '\\' + 'motion_information.json'))
                motion.post_processing_info = json.load(open(folder_path + '\\' + folder + '\\' + 'segmentation_information.json'))

                with open(folder_path + '\\' + folder + '\\' + subfolders + '\\' + file, 'r') as f:
                    json_file = json.load(f)
                    for key in list(json_file.keys()):
                        motion.add_datatype(key, json_file[key])

                full_data.append(motion)

    return full_data



if __name__ == '__main__':
    # data = return_data(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Test_Python')
    # data = return_data_with_names(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Test_Python')
    # data = data_gathering_dict(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed\Damien_1_Char00_SEGMENTED_TEEEEEEEEEST\NB_SEG_2')
    # data = data_gathering_full(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Test_Python')
    data = adhoc_gathering(r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Test_Python')
    pdb.set_trace()
