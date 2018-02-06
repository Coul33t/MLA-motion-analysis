from motion_classes.datatype import *

class Motion():
    def __init__(self, name='NONE'):
        self.name = name

        self.pre_processing_info = {}
        self.post_processing_info = {}

        # Dic str: Datatype()
        self.datatypes = {}

    def add_datatype(self, name, data):
        datatype = Datatype(name)
        datatype.joints = data
        self.datatypes[name] = datatype

    def get_joint_list(self):
        if self.datatypes:
            return self.datatypes[list(self.datatypes.keys())[0]].get_joint_list()
        else:
            return None

    def get_datatypes_names(self):
        return list(self.datatypes.keys())

    def get_datatype(self, name):
        if name in self.datatypes.keys():
            return self.datatypes[name]
