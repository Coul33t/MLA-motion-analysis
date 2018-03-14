import pdb

import os
import json

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

        for item, value in kwargs.items():
            # If it's a list of nouns, we merge it
            # e.g. ['Hand', 'Arm', 'Head']
            # ->    'Hand, Arm, Head'
            if isinstance(value, list) and isinstance(value[0], str):
                if len(value) == 1:
                    res_dict[item] = value[0]
                else:
                    res_dict[item] = ', '.join(value)


            # Else, it's nothing special (either a list or a dict or a value)
            else:
                res_dict[item] = value

            self.list_of_properties.add(item)

        self.result_list.append(res_dict)

    def get_res(self, **kwargs):

        # validate values in kwargs
        # Since the keys() method follows the set interface, we can do this
        # ((intersection of the sets == initial set) means that all the keys
        # of kwargs are present in the keys of the results)
        if not (kwargs.keys() & self.list_of_properties == kwargs.keys()):
            print('ERROR: {} not in results properties'.format(name))
            print('Available properties are:')
            for key in self.result_list[0].keys():
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
            valid = True

            for key, val in kwargs.items():
                if val[1] == 'eq':
                    if res[key] != val[0]:
                        valid = False
                        break

                elif val[1] == 'inf':
                    if val[0] < res[key] :
                        valid = False
                        break

                elif val[1] == 'sup':
                    if val[0] > res[key]:
                        valid = False
                        break

                elif val[1] == 'infeq':
                    if val[0] <= res[key]:
                        valid = False
                        break

                elif val[1] == 'supeq':
                    if val[0] >= res[key]:
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

    # TODO: redo (no adhoc data)
    def export_data(self, path, data_to_export='all', text_export=True, json_export=True):
        """
            Exports the data to a .txt file
        """
        if data_to_export == 'all':
            data_to_export = self.result_list

        if json_export:
            json_list = []

        for res in data_to_export:
            
            if json_export:
                json_list.append(res)

            if text_export:
                file_name_txt = 'output_' + res['data_used'].replace(" ", "").replace(",", "") + '.txt'   

                txt_mode = 'w'
            
                if os.path.exists(path + '//' + file_name_txt):
                    txt_mode = 'a'
               

                with open(path + '//' + file_name_txt, txt_mode) as f_txt:
                    if txt_mode == 'a':
                        f_txt.write('\n\n\n')

                    f_txt.write('Data used: ' + res['data_used'])
                    f_txt.write('\nk = ' + str(res['k']))
                    f_txt.write('\nJoints used: ' + res['joints_used'])

                    f_txt.write('\n-------------------------------------')
                    f_txt.write('\nGlobal inertia: ' + str(res['global_inertia']))
                    f_txt.write('\n-------------------------------------')

                    for c in res['per_cluster_inertia']:
                        f_txt.write('\n' + c + ' inertia: ' + str(res['per_cluster_inertia'][c]))

                    f_txt.write('\n-------------------------------------')

                    for m in res['metrics']:
                        f_txt.write('\n' + m + ': ' + str(res['metrics'][m]))

                    f_txt.write('\n-------------------------------------')

                    for i, c in enumerate(res['motion_repartition']):
                        if i != 0:
                            f_txt.write('\n')
                        f_txt.write('\nMotions in ' + c + ': ' + ', '.join(res['motion_repartition'][c]))

                    f_txt.write('\n-------------------------------------')

        if json_export and data_to_export:
            file_name_json = 'output_' + self.result_list[0]['data_used'].replace(" ", "").replace(",", "") + '.json'

            # By default, we create and write in the file
            json_mode = 'w'

            # If the file already exists, we append to it instead
            if os.path.exists(path + '//' + file_name_json):
                json_mode = 'a'

            with open(path + '//' + file_name_json, json_mode) as json_f:
                json.dump(json_list, json_f)


if __name__ == '__main__':
    res = Results()
    res.get_res(global_inertia=50)