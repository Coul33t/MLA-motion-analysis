from motion_classes.datatype import Datatype

class Motion():
    def __init__(self, name='NONE'):
        self.name = name

        self.pre_processing_info = {}
        self.post_processing_info = {}

        # Dic str: Datatype()
        self.datatypes = {}

        self.laterality = 'Ambidextrous'

    def add_datatype(self, name, data):
        datatype = Datatype(name)
        datatype.joints = data
        self.datatypes[name] = datatype

    def get_joint_list(self):
        if self.datatypes:
            joints_set = set()
            for _, datatypes in self.datatypes.items():
                for joint in datatypes.get_joint_list():
                    joints_set.add(joint)
            return list(joints_set)
        else:
            return None
    
    def get_joint_list_datatype(self, name):
        if self.datatypes:
            joints_set = set()
            for _, datatypes in self.datatypes.items():
                for joint in datatypes.get_joint_list():
                    joints_set.add(joint)
            return list(joints_set)
        else:
            return None

    def get_datatypes_names(self):
        return list(self.datatypes.keys())

    def get_datatype(self, name):
        if name in self.datatypes.keys():
            return self.datatypes[name]

    def validate_motion(self):
        print(f'Validating {self.name}...')

        if self.name == 'NONE':
            print('Motion has name NONE.')
            return False

        if not self.pre_processing_info:
            print('Pre-processing information is empty.')
            return False

        if not self.post_processing_info:
            print('Post-processing information is empty.')
            return False

        sorted_joint_list = sorted(self.pre_processing_info['joints names'])

        # For very joints there should be
        for joint in sorted_joint_list:
            # For every datatype
            for datatype in self.datatypes:

                datatype_joint_list = []
                # For every joints in the datatype
                for joint_datatype in self.datatypes[datatype].joints:

                    datatype_joint_list.append(joint_datatype)

                    if None in self.datatypes[datatype].get_joint_values(joint_datatype):
                        print(f'Joint {joint_datatype} from datatype {datatype} has None value.')
                        return False

                if sorted_joint_list != sorted(datatype_joint_list):
                    print(f'Some joints are different between original motion and datatype {datatype}.')
                    return False

        print(f'{self.name} validated.')
        return True


