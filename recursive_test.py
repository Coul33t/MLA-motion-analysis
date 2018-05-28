import pdb

import numpy as np

from data_import import json_import

from main import (joint_selection,
                  data_selection)

from algos.kmeans_algo import (kmeans_algo,
                               adjusted_rand_score_computing,
                               silhouette_score_computing)

from misc.tree import (Tree, Node, DFS, BFS)

SS_THRESHOLD = 0.5

ITERATION_COUNTER = 0

def init():
    """
        Initialise the data, and launch the recursive clustering
    """
    original_data = []
    original_data = json_import(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/Throw_ball_less/', 'Leo')
    datatypes_used = [['BegMaxEndSpeedNorm'], ['BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz'], ['BegMaxEndSpeedNorm', 'BegMaxEndSpeedDirx', 'BegMaxEndSpeedDiry', 'BegMaxEndSpeedDirz']]
    joints = [['RightHand']]
    tree = Tree()
    root = Node()
    value = {}
    value['ss'] = 1
    value['datatypes'] = datatypes_used
    value['joints'] = joints
    value['datanames'] = [x.name for x in original_data]
    root.value = value
    tree.add_node(root)

    recursive_clustering(original_data, 'all', datatypes_used, joints, tree, root)
    pdb.set_trace()

def recursive_clustering(original_data, data_to_keep, datatype_used, joints, tree, pnode):
    """
        Performs a recursive clustering, as long as SS > THRESHOLD
        and sample_n > 2
    """

    # TODO: something without this (because there'll be a huge memory
    # consumption)
    global ITERATION_COUNTER
    ITERATION_COUNTER += 1
    sub_data = []
    if data_to_keep == 'all':
        sub_data = original_data
    else:
        for motion in original_data:
            if motion.name in data_to_keep:
                sub_data.append(motion)

    for data in datatype_used:
        for joint in joints:

            max_k_value = 10

            if len(sub_data) <= max_k_value:
                max_k_value = len(sub_data) - 1

            # No need to compute a k-means for a sample number = 2
            if max_k_value <= 2:
                continue

            for k in range(2, max_k_value + 1):

                # We keep the joints' data we're interested in
                # (from the motion class)
                selected_data = joint_selection(sub_data, joint)

                # We select the data we want and we put them in the right shape
                # for the algorithm
                # [sample1[f1, f2, ...], sample2[f1, f2, f3...], ...]
                features = data_selection(selected_data, data)

                # K-means
                res = kmeans_algo(features, k=k)

                # Compute desired metric
                s_score = silhouette_score_computing(features, res.labels_)
                # print(f'Data length: {len(res.labels_)}')
                # print(f'Clusters number: {np.unique(res.labels_)}')
                # print(f'Clusters size:')
                # for c in range(len(np.unique(res.labels_))):
                #     print(f'c{c}: {len(np.where(res.labels_ == c)[0])}')
                print(f'SS: {s_score}')

                 # If it's value > threshold:
                if s_score < SS_THRESHOLD:
                    print(f'TRUTH HURTS')
                else:
                    print(f'SIGMAR BLESS THIS RAVAGED SS')
                    idx_c = []

                    # We get the indexes of motion in every clusters
                    for c in range(len(np.unique(res.labels_))):
                        idx_c.append(np.where(res.labels_ == c)[0].tolist())

                    cnode = Node()
                    value = {}
                    value['data'] = data
                    value['joint'] = joint
                    value['ss'] = s_score
                    for i,rep in enumerate(idx_c):
                        value[f'c{i}'] = [x.name for i, x in enumerate(original_data) if i in idx_c[0]]
                    cnode.value = value
                    cnode.set_parent(pnode)
                    tree.add_node(cnode)
                    # Recursive clustering on each cluster
                    for rep in idx_c:
                        # If we have less than 3 samples in a cluster,
                        # no need to perfom clustering on it
                        if len(rep) < 3:
                            print(f'AN EMPTY CLUSTER SERVES NO ONE')
                            return
                        else:
                            names = [x.name for i, x in enumerate(original_data) if i in idx_c[0]]
                            recursive_clustering(original_data, names, datatype_used, joints, tree, cnode)






if __name__ == '__main__':
    init()
