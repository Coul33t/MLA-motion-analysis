import numpy as np
import sys
from tools import flatten_list

def data_gathering(data, datatype_joints: dict):
    """
    """

    # data_joint is a dict:
    #     {'datatype'(str): ['joint1', 'joint2'](list)}

    # return is:
    # [[sample(1)], [sample(2)], ..., [sample(n)]] -> 1 sample = 1 motion
    #
    # sample is:
    # [j1f1, j1f2, j1f3, ..., j1fm,
    #  j2f1, j2f2, j2f3, ..., j2fm,
    #  ...,
    #  jnf1, jnf2, jnf3, ..., jnfm]
    #
    # return is:
    # [[j1f1, j1f2, j1f3, ..., j1fm, j2f1, j2f2, j2f3, ..., j2fm, ..., jnf1, jnf2, jnf3, ..., jnfm], -> MOTION 1
    #  [j1f1, j1f2, j1f3, ..., j1fm, j2f1, j2f2, j2f3, ..., j2fm, ..., jnf1, jnf2, jnf3, ..., jnfm]] -> MOTION 2


    # Since I'm not 100% used to numpy array, it'll be
    # converted to it right before the return
    features = []

    # for each motion, extract the same values
    for motion in data:
        tmp_features_list = []
        # for each required datatype/joints combination for the current motion:
        for datatype, joints in datatype_joints.items():
            # for each joint, get the datatypes
            if isinstance(joints, list):
                for joint in joints:
                    # get the joints values for a specified datatype
                    if joint['laterality']:
                        if motion.laterality == 'Left':
                            joint['joint'] = joint['joint'].replace('Right', motion.laterality)
                        elif motion.laterality == 'Right':
                            joint['joint'] = joint['joint'].replace('Left', motion.laterality)
                    try:
                        tmp_features_list.append(motion.get_datatype(datatype).get_joint_values(joint['joint']))
                    except AttributeError:
                        print(f'ERROR: {datatype} not present in {motion.name}.')
                        sys.exit()

            else:
                if joints['laterality']:
                    if motion.laterality == 'Left':
                        joints['joint'] = joints['joint'].replace('Right', motion.laterality)
                    elif motion.laterality == 'Right':
                        joints['joint'] = joints['joint'].replace('Left', motion.laterality)

                tmp_features_list.append(motion.get_datatype(datatype).get_joint_values(joints['joint']))

        features.append(flatten_list(tmp_features_list))

    return np.asarray(features)