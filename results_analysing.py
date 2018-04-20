import pdb

import numpy as np
import os
import json

from tools import (string_length_shortening,
                   string_redundancy_remover)

class Results:
    def __init__(self):
        # Will contain a list of dict
        self.result_list = []
        self.list_of_properties = set()

    def add_values(self, **kwargs):
        """
            Appends a dict containing the information and results of an 
            algorithm's pass
        """
        res_dict = {}

        for key, value in kwargs.items():
            # If it's a list of nouns, we merge it
            # e.g. ['Hand', 'Arm', 'Head']
            # ->    'Hand, Arm, Head'
            if isinstance(value, list) and isinstance(value[0], str):
                if len(value) == 1:
                    res_dict[key] = value[0]
                else:
                    res_dict[key] = ', '.join(value)


            # Else, it's nothing special (either a list or a dict or a value)
            else:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                res_dict[key] = value

            self.list_of_properties.add(key)

            if isinstance(value, dict):
                for key2 in value.keys():
                    self.list_of_properties.add(key2)

        self.result_list.append(res_dict)

    def get_res(self, **kwargs):

        # validate values in kwargs
        # Since the keys() method follows the set interface, we can do this
        # ((intersection of the sets == initial set) means that all the keys
        # of kwargs are present in the keys of the results)
        if not (kwargs.keys() & self.list_of_properties == kwargs.keys()):
            print('ERROR: {} not in results properties'.format(
                  kwargs.keys() - kwargs.keys() & self.list_of_properties))
            print('Available properties are:')
            for key, val in self.result_list[0].items():
                print('- {}'.format(key))
            return False

        # validating the values sent to kwargs
        # for each key, value should be [value_to_test, comparison operator]
        for key, val in kwargs.items():
            if not isinstance(val, list):
                print('ERROR: key \'{}\', value should be a list (the value and the comparison operator [eq, inf, sup, infeq, supeq])'.format(key))
                return False
            if (not isinstance(val[1], str)) or (val[1] not in ['eq', 'inf', 'sup', 'infeq', 'supeq']):
                print('ERROR: key \'{}\', 2nd argument of value should be str (either [eq, inf, sup, infeq, supeq])'.format(key))

        res_return = []

        for res in self.result_list:
            # Must use a tmp_res, see below (l.82)
            tmp_res = res
            valid = True

            for key, val in kwargs.items():

                # It means that the key is in a dictionnary or doesn't exist
                if key not in tmp_res.keys():
                    key_found_in_subdict = False
                    # We iterate over the key/values
                    for key_inner_dict, val_inner_dict in tmp_res.items():
                        # If the value is a dict, we try to find the original key inside it
                        if isinstance(val_inner_dict, dict):
                            if key in val_inner_dict.keys():
                                # In this case, if we do not use a tmp_res, the appended
                                # value will be just the dict
                                # (original line was ' res = val_inner_dict ')                        
                                tmp_res = val_inner_dict
                                key_found_in_subdict = True

                    if not key_found_in_subdict:
                        valid = False
                        break

                if val[1] == 'eq':
                    if val[0] != tmp_res[key]:
                        valid = False
                        break

                elif val[1] == 'inf':
                    if val[0] < tmp_res[key] :
                        valid = False
                        break

                elif val[1] == 'sup':
                    if val[0] > tmp_res[key]:
                        valid = False
                        break

                elif val[1] == 'infeq':
                    if val[0] <= tmp_res[key]:
                        valid = False
                        break

                elif val[1] == 'supeq':
                    if val[0] >= tmp_res[key]:
                        valid = False
                        break

            if valid:
                res_return.append(res)

        return res_return

    # TODO: redo (no adhoc data)
    def print_data(self, results_sublist):
        """
            Print all the results to the console
            WARNING: lot of text
        """
        
        for res in results_sublist:
            print('Data used: ' + res['data_used'])
            print('\nk = ' + str(res['k']))
            print('\nJoints used: ' + res['joints_used'])

            print('\n-------------------------------------')
            print('\nGlobal inertia: ' + str(res['global_inertia']))
            print('\n-------------------------------------')

            for c in res['per_cluster_inertia']:
                print('\n' + c + ' inertia: ' + str(res['per_cluster_inertia'][c]))

            print('\n-------------------------------------')

            for m in res['metrics']:
                print('\n' + m + ': ' + str(res['metrics'][m]))

            print('\n-------------------------------------')

            for i, c in enumerate(res['motion_repartition']):
                if i != 0:
                    print('\n')
                print('\nMotions in ' + c + ': ' + ', '.join(res['motion_repartition'][c]))

            print('\n-------------------------------------')

    def export_data(self, path, data_to_export='all', text_export=True, json_export=True):
        """
            Exports the data to a .txt file
        """

        # If the parameter data_to_export is empty, we export nothing
        # (e.g. get_res was called but returned no values)
        if len(data_to_export) == 0:
            return

        if data_to_export == 'all':
            data_to_export = self.result_list

        if json_export:
            json_list = []

        # Making up a name for the file and opening it
        if text_export:
            file_name_txt = 'output_' + self.result_list[0]['data_used'].replace(" ", "").replace(",", "")
            file_name_txt = string_redundancy_remover(file_name_txt) + '.txt'

            txt_mode = 'w'
            if os.path.exists(path + file_name_txt):
                txt_mode = 'a'
            else:
                # If the directory does not exists
                if not os.path.exists(path):
                    os.makedirs(path)
            
            f_txt = open(path + file_name_txt, txt_mode)

        # For every result
        for res in data_to_export:
            
            # For JSON, we export them at the end, so we just append what to export
            if json_export:
                json_list.append(res)

            # For text export, we format it to be a bit cleaner
            if text_export:
                
                if txt_mode == 'a':
                    f_txt.write('\n\n\n')
               
                f_txt.write('\n-------------------------------------')

                # For everything in result
                for key, values in res.items():

                    # If it's a dic
                    if isinstance(values, dict):
                        # We replace the "_" with " " for the keys values (more readable)
                        f_txt.write('\n{}:'.format(key.replace("_", " ")))
                        
                        # Then, for each values inside the dic
                        for inside_key, inside_values in values.items():
                            # Casting numerical values to string
                            if isinstance(inside_values, int):
                                inside_values = str(inside_values)
                            # Same formatting as above  
                            f_txt.write("\n\t{}: {}".format(inside_key.replace("_", " "), inside_values))
                    # Else, no formating necessary
                    else:
                        # Casting numerical values to string
                        if isinstance(values, int):
                            values = str(values)
                        # Same formatting as above
                        f_txt.write('\n{}: {}'.format(key.replace("_", " "), values))


                f_txt.write('\n-------------------------------------')

        # If it was text export, we close the file since we're done with it
        if text_export:
            f_txt.close()
            
        # For JSON export
        if json_export and data_to_export:
            file_name_json = 'output_' + self.result_list[0]['data_used'].replace(" ", "").replace(",", "")
            file_name_json = string_redundancy_remover(file_name_json) + '.json'

            # By default, we create and write in the file
            json_mode = 'w'

            # If the file already exists, we append to it instead
            if os.path.exists(path + file_name_json):
                json_mode = 'a'

            else:
                # If the directory does not exists
                if not os.path.exists(path):
                    os.makedirs(path)

            # We dump it inside. Yup, it's as simple as that.
            with open(path + file_name_json, json_mode) as json_f:
                json.dump(json_list, json_f)


if __name__ == '__main__':
    res = Results()
    res.get_res(global_inertia=50)