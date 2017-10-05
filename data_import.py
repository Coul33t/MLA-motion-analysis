from tools import file_name_gathering, natural_keys
from collections import OrderedDict

def data_gathering_dict(folder_path, joints_to_append=None):
    """
        Put the data from multiples files (in the folder folder_path) into a dictionnary.
        joints_to_append is there if you want only certains joints.
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