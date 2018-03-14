import pdb

import os
import json

class Results:
    def __init__(self):
        # Will contain a list of dict
        self.result_list = []
        self.list_of_properties = []

    def add_values(self, k, data_used, joints_used, global_inertia, 
                   per_cluster_inertia, metrics, motion_repartition):
        """
            Appends a dict containing the information and results of an 
            algorithm's pass
        """
        res_dict = {}
        res_dict['k'] = k

        if isinstance(data_used, list):
            if len(data_used) == 1:
                res_dict['data_used'] = data_used[0]
            else:
                res_dict['data_used'] = ', '.join(data_used)

        if isinstance(joints_used, list):
            joints_used = ', '.join(joints_used)

        res_dict['joints_used'] = joints_used
        res_dict['global_inertia'] = global_inertia
        
        c_inertia_values = {}

        # per_cluster_inertia is a list
        for i, inertia_c in enumerate(per_cluster_inertia):
            c_inertia_values['c' + str(i)] = inertia_c

        res_dict['per_cluster_inertia'] = c_inertia_values

        # metrics is a dict
        res_dict['metrics'] = metrics

        # motion_repartition is a dict
        res_dict['motion_repartition'] = motion_repartition

        self.result_list.append(res_dict)

    def get_res_global_inertia(self, global_inertia_threshold):
        """
            Returns all the results which have a global inertia equals or lower to
            the threshold
        """
        # Beautiful one-liner boi
        return [res for res in self.result_list if res['global_inertia'] >= global_inertia_threshold]

    #TODO: use kwargs to get a few arguments to seek the desired values in self.result_list
    def get_res(self, **kwargs):

        pdb.set_trace()
        # validate values in kwargs
        # Since the keys() method follows the set interface, we can do this
        # (intersection of the sets == initial set means that all the keys
        # of kwargs are present in the keys of the results)
        if not (kwargs.keys() & self.result_list[0].keys() == kwargs.keys())
            print('ERROR: {} not in results properties'.format(name))
            print('Available properties are:')
            for key in self.result_list[0].keys():
                print('- {}'.format(key))
            return False

        res_return = []
            valid = True

            for key, val in kwargs.items():



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

    def export_data(self, path, text_export=True, json_export=True):
        """
            Exports the data to a .txt file
            TODO: add **kwargs to export specific results
        """
        if json_export:
            json_list = []

        for res in self.result_list:
            
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


        if json_export:
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