import numpy as np

from fastdtw import fastdtw

from data_import import json_import

from collections import OrderedDict

from tools import flatten_list

import data_labels as dl

import pdb


def custom_dst(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

def custom_dst2(x, y):
    return abs(x-y)

def joint_selection(data, joints_to_append):
    """
        This function returns a list of desired joints data, from a list of
        Motion objects.
    """

    # We gonna switch from dict to OrederedDict, to ensure that each
    # features vector has the same joints order
    selected_data = []

    datatypes = data[0].get_datatypes_names()

    # If joints_to_append isn't specified, we take all the joints. Just to be
    # sure that the order won't change between 2 motion dicts, we take the keys
    # of one motion and use it as the reference for insertion order
    if not joints_to_append:
        joints_to_append = data[0].get_joint_list()

    for motion in data:
        # We keep the name
        selected_joints_motion = [motion.name, OrderedDict()]

        # For each datatype
        for datatype in datatypes:

            # We use an OrderedDict. Since we iterate over the joints_to_append
            # list, the order is ensured to be the same for each motion
            joints_selected = OrderedDict()

            # For each joint to append
            if isinstance(joints_to_append, list):
                for joint in joints_to_append:
                    joints_selected[joint] = motion.get_datatype(datatype).get_joint_values(joint)

            else:
                joints_selected[joints_to_append] = motion.get_datatype(datatype).get_joint_values(joints_to_append)

            selected_joints_motion[1][datatype] = joints_selected

        selected_data.append(selected_joints_motion)

    return selected_data


def data_selection(data, data_to_keep):
    """
        This function returns an array of desired features.
    """

    features = []

    # For each motion
    # motion[0] is the filename
    # motion[1] are the values, ordered by joints and datatypes
    # (datatype->joints->values)
    for motion in data:
        motion_feat = OrderedDict()

        # For each data type we want to gather
        for datatype in data_to_keep:

            # Don't ask how it works (= nb_frames - 1 for speed)
            nb_values = len(list(motion[1][datatype].values())[0])

            for i in range(nb_values):
                # For each joint
                for joint in motion[1][datatype]:
                    # If it doesn't exists, create an empty list then append
                    # Else, append

                    motion_feat.setdefault(joint, []).append(motion[1][datatype][joint][i])

        features.append(flatten_list(list(motion_feat.values())))

    # We return the list as a numpy array, as it is more
    # convenient to use
    return np.asarray(features)


def dtw_compute(data):
    output = []

    if isinstance(data, dict):

        # Too lazy to figure a way to do it with dict
        # So we gonna end up with a beautiful symmetric matrix
        # EDIT: figured it out (lel)

        for i, (k, v) in enumerate(data.items()):

            if not isinstance(v, np.ndarray):
                v = np.asarray(v)
            v.reshape(-1, 1)

            output.append([])

            for j, (k2, v2) in enumerate(data.items()):
                if j > i:
                    if not isinstance(v2, np.ndarray):
                        v2 = np.asarray(v2)
                    v2.reshape(-1, 1)

                    dist, path = fastdtw(v, v2, dist=custom_dst2)
                    output[i].append(dist)

        return output

    elif isinstance(data, list) or isinstance(data, np.ndarray):
        for i, d in enumerate(data):
            print(f'signal {i}')
            x = d
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            x.reshape(-1, 1)

            output.append([])

            for j, d2 in enumerate(data):
                if j > i:
                    y = d2
                    if not isinstance(y, np.ndarray):
                        y = np.asarray(y)
                    y.reshape(-1, 1)

                    dist, path = fastdtw(x, y, dist=custom_dst2)
                    output[i].append(dist)
                else:
                    output[i].append(None)

        return output


def main_fastdtw_compute():
    original_data = []
    original_data = json_import(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/Throw_ball/', 'Leo')

    if not original_data:
        print('ERROR: no data found (check your names).')

    data_to_select = [['SpeedNorm']]
    # If there's no specific datatype defined, we take all the data available
    if not data_to_select:
        data_to_select = set([name for motion in original_data for name in motion.datatypes])

    print('\nData used: {}'.format(data_to_select))

    joint_to_use = ['RightHand']
    # If there's no joint to select, then we take all of them
    if joint_to_use is None:
        joint_to_use = original_data[0].get_joint_list()

    print('Joints used: {}\n'.format(joint_to_use))

    # For each datatype
    for data in data_to_select:
        print(f'Running for {data}')

        # For each joint combination
        for joint in joint_to_use:
            joint_name = joint

            print(f'Running for {joint_name}')

            if isinstance(joint, list):
                joint_name = ','.join(joint)


            # We keep the joints' data we're interested in (from the motion class)
            selected_data = joint_selection(original_data, joint)

            # We select the data we want and we put them in the right shape
            # for the algorithm [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
            features = data_selection(selected_data, data)

            # If we only work with the succesful motions,
            # we only keep these ones (c == 1 for success)
            mat = dtw_compute(list(features))

            if isinstance(joint, list):
                joint = '_'.join(joint)
            if isinstance(data, list):
                data = '_'.join(data)

            f_o = open(joint + '_' + data + "_Leo_fastdtw_matrix.txt", "w")
            for line in mat:
                for val in line:
                    if val == None:
                        f_o.write('- ')
                    else:
                        f_o.write(str(float(val)) + ' ')
                f_o.write('\n')
            f_o.close()

def main_regroup(verbose=False):
    f_o = open("RightHand_SpeedNorm_Leo_fastdtw_matrix.txt", "r")
    txt = f_o.read()
    mat = []
    for i,line in enumerate(txt.split('\n')):
        if line:
            mat.append([])
            for val in line.split():
                if val == '-':
                    mat[i].append(None)
                else:
                    mat[i].append(float(val))



    success = dl.LEO_THROW_TYPES
    idx_succ = [i for i, l in enumerate(success) if l == 1]
    idx_fail = [i for i, l in enumerate(success) if l == 0]

    succ_array = []
    fail_array = []

    #success
    for i,idx_row in enumerate(idx_succ):
        for j,idx_col in enumerate(idx_succ):
            if j > i:
                succ_array.append(mat[idx_row][idx_col])

    #fails
    for i,idx_row in enumerate(idx_fail):
        for j,idx_col in enumerate(idx_fail):
            if j > i:
                fail_array.append(mat[idx_row][idx_col])

    print('---------------------------------------------------------------------')
    print(f'Mean distance of success motion between them: {np.mean(succ_array)}')
    print(f'Min distance of success motion between them: {min(succ_array)}')
    print(f'Max distance of success motion between them: {max(succ_array)}')
    print('---------------------------------------------------------------------')
    print(f'Mean distance of failure motion between them: {np.mean(fail_array)}')
    print(f'Min distance of failure motion between them: {min(fail_array)}')
    print(f'Max distance of failure motion between them: {max(fail_array)}')
    print('---------------------------------------------------------------------')

    succ_dst_other = []
    fail_dst_other = []

    for i in range(100):
        # If it's a fail, we compare it to the success
        if success[i] == 0:
            dst = []
            for idx in idx_succ:
                if mat[i][idx] != None:
                    dst.append(mat[i][idx])
                else:
                    dst.append(mat[idx][i])
            fail_dst_other = np.mean(dst)
            if verbose:
                print(f'Mean distance sample {i} (failed) from other: {np.mean(dst)}')
                print('---------------------------------------------------------------------')

        # If it's a success, we compare it to the fails
        elif success[i] == 1:
            dst = []
            for idx in idx_fail:
                if mat[i][idx] != None:
                    dst.append(mat[i][idx])
                else:
                    dst.append(mat[idx][i])
            succ_dst_other = np.mean(dst)
            if verbose:
                print(f'Mean distance sample {i} (success) from other: {np.mean(dst)}')
                print('---------------------------------------------------------------------')

    print('---------------------------------------------------------------------')
    print(f'Mean distance from success to other: {np.mean(succ_dst_other)}')
    print(f'Mean distance from failure to other: {np.mean(fail_dst_other)}')
    print('---------------------------------------------------------------------')
    pdb.set_trace()

if __name__ == '__main__':
    main_fastdtw_compute()
    main_regroup()
