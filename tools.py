import os
import re

from collections import OrderedDict

def file_name_gathering(folder_path):
    """
        Collects the file name from folder_path.
    """
    return  next(os.walk(folder_path))[2]

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split('(\d+)', text)]

def flatten_list(list_to_flatten):
    return [item for sublist in list_to_flatten for item in sublist]

def dic_to_list(data_dict):
    return [list(item.values()) for item in data_dict]

def merge_dict(list_of_dict):
    """
        Merge a few dict into a new one
    """

    new_dic = OrderedDict()

    for dic in list_of_dict:
        # Empty dicts evaluate to false
        if not new_dic:
            # First dict dictates the key order
            for key in dic.keys():
                new_dic[key] = dic[key]

        else:
            # Appending values
            for key in new_dic.keys():
                new_dic[key].extend(dic[key])

    return new_dic


def select_joint(dic, joints_to_select):

    selected_joints = OrderedDict()

    for key in dic.keys():
        if key in joints_to_select:
            selected_joints[key] = dic[key]

    return selected_joints


def flatten_data_struct(data):
    """
        flatten the data struct extracted by return_data
        into something like that : [motion list[joint OrderedDict (full motion)]]
    """

    # List of list (seg)
    for motion in data:
        new_dic = OrderedDict()

        # List of dic (speed)
        for seg in motion:

            for dic in seg:
                # Empty dicts evaluate to false
                if not new_dic:
                    # First dict dictates the key order
                    for key in dic.keys():
                        new_dic[key] = dic[key]

                else:
                    pass




def motion_dict_to_list(data_dict):
    """
                       j1    f1, f2, f3    j2    f1, f2, f3    j3    f1, f2, f3
        Take a dict {'val1': (1,10,100), 'val2': (2,20,200), 'val3': (3,30,300)}
                                         f1        f2            f3
        and transform it into a list [(1,2,3), (10,20,30), (100,200,300)].

        Used to transform the dictionnary extracted from csv files
        into a coherent, processable data structure.
    """
    llist = list(data_dict.values())
    new_list = []

    for i in range(len(llist[0])):
        new_list.append([item[i] for item in llist])

    return new_list
