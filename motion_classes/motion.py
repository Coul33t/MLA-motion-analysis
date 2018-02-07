from motion_classes.datatype import *

#TODO: add joints name to the json file (C++)
class Motion():
    def __init__(self, name='NONE'):
        self.name = name

        self.pre_processing_info = {}
        self.post_processing_info = {}

        # Dic str: Datatype()
        self.datatypes = {}

        # List[str]
        self.joints_list = []

    def add_datatype(self, name, data):
        datatype = Datatype(name)
        datatype.joints = data
        self.datatypes[name] = datatype

        if not self.joints_list:
            self.joints_list = datatype.get_joint_list()

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

    def validate_motion(self):
        print('Validating {}...'.format(self.name))
        passed = True

        if self.name == 'NONE':
            print('Motion has name NONE.')
            passed = False

        if not self.pre_processing_info:
            print('Pre-processing information is empty.')
            passed = False

        if not self.post_processing_info:
            print('Post-processing information is empty.')
            passed = False

        
        sorted_joint_list = sorted(self.joints_list)

        # For very joints there should be
        for joint in sorted_joint_list:
            # For every datatype
            for datatype in self.datatypes:
                
                datatype_joint_list = []
                # For every joints in the datatype
                for joint_datatype in self.datatypes[datatype].joints:

                    datatype_joint_list.append(joint_datatype)

                    if None in self.datatypes[datatype].get_joint_values(joint_datatype):
                        print('Joint {} from datatype {} has None value.'.formtat(joint_datatype, datatype))
                        passed = False

                if sorted_joint_list != sorted(datatype_joint_list):
                    print('Some joints are different between original motion and datatype {}.'.format(datatype))
                    passed = False

        if passed == True:
            print('{} validated.'.format(self.name))
        return passed


