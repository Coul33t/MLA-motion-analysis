import numpy as np

def data_gathering(std_data, exp_data, datatype_joints: dict):
    """
    """

    # data_joint is a dict:
    #     {'datatype'(str): ['joint1', 'joint2'](list)}

    # return is:
    # [[sample(1)], [sample(2)], ..., [sample(n)]] -> 1 sample = 1 motion
    #
    # sample is:
    # {joint_name_1 datatype_name_1: val, ..., joint_name_1 datatype_name_m: val,
    #  joint_name_2 datatype_name_1: val, ..., joint_name_2 datatype_name_m: val,
    #  ...,
    #  joint_name_n datatype_name_1: val, ..., joint_name_n datatype_name_m: val,]
    #
    # return is:
    # [[j1f1, j1f2, j1f3, ..., j1fm, j2f1, j2f2, j2f3, ..., j2fm, ..., jnf1, jnf2, jnf3, ..., jnfm], -> MOTION 1
    #  [j1f1, j1f2, j1f3, ..., j1fm, j2f1, j2f2, j2f3, ..., j2fm, ..., jnf1, jnf2, jnf3, ..., jnfm]] -> MOTION 2


    # Since I'm not 100% used to numpy array, it'll be
    # converted to it right before the return
    features = {}

    # for each required datatype/joints combination for the current motion:
    for datatype, joints in datatype_joints.items():
        # for each joint, get the datatypes
        for joint in joints:
            # get the joints values for a specified datatype
            features[f'{joint} {datatype}'] = {'student': std_data.get_datatype(datatype).get_joint_values(joint),
                                               'expert': exp_data.get_datatype(datatype).get_joint_values(joint)}

    return features

def compare(features):
    compared = {}
    for joint_and_datatype, values in features.items():
        std_val = np.asarray(values['student'])
        exp_val = np.asarray(values['expert'])
        compared[joint_and_datatype] = std_val - exp_val
    return compared





