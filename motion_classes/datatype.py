class Datatype:
    def __init__(self, name='DEFAULT DATATYPE NAME'):
        self.name = name
        # Dic, str: list[values]
        self.joints = {}

    def get_joint_list(self):
        if self.joints:
            return list(self.joints.keys())
        else:
            return None

    def get_joint(self, name):
        if name in self.joints.keys():
            return self.joints[name]

    def get_joint_values(self, name):
        if name in self.joints.keys():
            return self.joints[name]