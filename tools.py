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
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def flatten_list(list_to_flatten):
    return [item for sublist in list_to_flatten for item in sublist]

def dic_to_list(data_dict):
    return [list(item.values()) for item in data_dict]

def motion_dict_to_list(data_dict):
    """
        Take a dict {'val1': (1,10,100), 'val2': (2,20,200), 'val3': (3,30,300)}
        and transform it into a list [(1,2,3), (10,20,30), (100,200,300)].

        Used to transform the dictionnary extracted from csv files
        into a coherent, processable data structure.
    """
    llist = list(data_dict.values())
    new_list = []

    for i in range(len(llist[0])):
        new_list.append([item[i] for item in llist])

    return new_list