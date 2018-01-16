class Motion:
    def __init__(self, path_to_file, json_info=None):
        self.dic = {}
        self.path_to_file = path_to_file
        self.info = json_info

    # Merge dicts. dict_or_dicts can be:
    #     - An OrderedDict  
    #     - A list of OrderedDict
    def add_values_to_dict(self, dict_or_dicts):
        if type(dict_or_dicts) == OrderedDict:
            if not self.dic:
                # First dict dictates the key order
                for key in dict_or_dicts.keys():
                    self.dic[key] = dict_or_dicts[key]

            else:
                # Appending values
                for key in self.dic.keys():
                    self.dic[key].extend(dict_or_dicts[key])

        if type(dict_or_dicts) == list:
            for dic in dict_or_dicts:
                # Empty dicts evaluate to false
                if not self.dic:
                    # First dict dictates the key order
                    for key in dic.keys():
                        self.dic[key] = dic[key]

                else:
                    # Appending values
                    for key in self.dict.keys():
                        self.dic[key].extend(dic[key])

    def get_joints_name(self):
        return list(self.dict.keys())

    def get_joint_values(self, joint):
        if joint in self.dict.keys():
            return self.dict[joint]
        else:
            print('{} not in joints.'.format(joint))
            return None

    def get_size(self):
        return len(self.dict)

    def get_info(self, param=None):
        if not param:
            return self.info
        else:
            if param in self.info.keys():
                return self.info[param]
            else:
                print('{} not in motion infos.'.format(param)) 
                return None
