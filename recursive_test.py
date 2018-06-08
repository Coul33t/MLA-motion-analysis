import pdb

import numpy as np

import igraph
import networkx as nx

from data_import import json_import

from main import (joint_selection,
                  data_selection)

from algos.kmeans_algo import (kmeans_algo,
                               adjusted_rand_score_computing,
                               silhouette_score_computing)

import data_labels as dl

SS_THRESHOLD = 0.5

ITERATION_COUNTER = 0

def select_one_class(original_data, labels, c_to_keep):
    new_original_data = []

    for i, motion in enumerate(original_data):
        if labels[i] == c_to_keep:
            new_original_data.append(motion)

    return new_original_data

def motion_classes(original_data, labels):
    motion_with_classes = []

    for i, motion in enumerate(original_data):
        motion_with_classes.append([motion.name, labels[i]])

    return motion_with_classes

def compute_clusters(labels, gt):

    n_labels = labels
    n_gt = [x[1] for x in gt]

    if not isinstance(n_labels, np.ndarray):
        n_labels = np.array(n_labels)
    if not isinstance(n_gt, np.ndarray):
        n_gt = np.array(n_gt)

    # We don't know if c0 (labels) == c0 (gt), so we take the labeling
    # that gives the highest good clustering

    score = (len(n_labels) - sum(abs(n_labels - n_gt)))/len(n_labels)

    # If the best value is for the reversed labels, we do it
    if (1 - score > score):
        n_gt = abs(n_gt - 1)

    idx_gt_0 = np.where(n_gt == 0)
    idx_labels_0 = np.where(n_labels == 0)

    idx_gt_1 = np.where(n_gt == 1)
    idx_labels_1 = np.where(n_labels == 1)

    c1_together = 0
    if len(idx_gt_0[0]) > 0:
        c1_together = len(np.intersect1d(idx_gt_0, idx_labels_0)) / len(idx_gt_0[0])

    c2_together = 0
    if len(idx_gt_1[0]) > 0:
        c2_together = len(np.intersect1d(idx_gt_1, idx_labels_1)) / len(idx_gt_1[0])

    return [c1_together, c2_together]


def init(c_to_keep=False, debug=False):
    """
        Initialise the data, and launch the recursive clustering
    """
    global ITERATION_COUNTER

    original_data = []
    original_data = json_import(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/Throw_ball/', 'Leo')
    datatypes_used = [['BegMaxEndSpeedNorm'], ['BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz'], ['BegMaxEndSpeedNorm', 'BegMaxEndSpeedDirx', 'BegMaxEndSpeedDiry', 'BegMaxEndSpeedDirz']]
    datatypes_used = [['BegMaxEndSpeedNorm'], ['BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz']]
    joints = [['RightHand']]

    original_data_classes = motion_classes(original_data, dl.LEO_LABELS_2)

    # if c_to_keep:
    #     original_data = select_one_class(original_data, dl.LEO_THROW_LABELS, c_to_keep)


    # +1 for the root
    i_graph = igraph.Graph()
    value = {}
    value['ss'] = 1
    value['datatypes'] = datatypes_used
    value['joints'] = joints
    value['datanames'] = [x.name for x in original_data]
    i_graph.add_vertex('root', **value)

    p_id = [0]
    recursive_clustering(original_data, 'all', datatypes_used, joints, i_graph, p_id, debug=debug)

    # Assigning colours to the vertices (dict is pretty self-explanatory)
    color_dict = {'black': [0, 0.25], 'red':[0.25, 0.5], 'orange': [0.5, 0.75], 'green': [0.75, 1]}
    for i,vertex in enumerate(i_graph.vs):
        # vertex['label'] = i
        if vertex['ss'] == 1:
            vertex['color'] = 'blue'
        else:
            for k, v in color_dict.items():
                if v[0] <= vertex['ss'] < v[1]:
                    vertex['color'] = k

    bbox = (1000, 1000)
    igraph.plot(i_graph, "output_graph_tree.svg", layout=i_graph.layout('tree'), vertex_size=5, bbox=bbox)
    igraph.plot(i_graph, "output_graph_large.svg", layout=i_graph.layout('large_graph'), vertex_size=5, bbox=bbox)
    igraph.plot(i_graph, "output_graph_circular.svg", layout=i_graph.layout('rt_circular'), vertex_size=5, bbox=bbox)
    i_graph.save("output.gml", format="gml")
    i_graph.save("output.gml", format="graphml")

    if debug:
        print(f'Iterations: {ITERATION_COUNTER}')

def init_test_stop(c_to_keep=False, debug=False):
    """
        Initialise the data, and launch the recursive clustering
    """
    global ITERATION_COUNTER

    original_data = []
    original_data = json_import(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/Throw_ball/', 'Leo')
    datatypes_used = [['BegMaxEndSpeedNorm'], ['BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz'], ['BegMaxEndSpeedNorm', 'BegMaxEndSpeedDirx', 'BegMaxEndSpeedDiry', 'BegMaxEndSpeedDirz']]
    datatypes_used = [['BegMaxEndSpeedNorm'], ['BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz']]
    joints = [['RightHand']]

    original_data_classes = motion_classes(original_data, dl.LEO_LABELS_2)


    # +1 for the root
    i_graph = igraph.Graph()
    value = {}
    value['ss'] = 1
    value['datatypes'] = datatypes_used
    value['joints'] = joints
    value['datanames'] = [x.name for x in original_data]
    i_graph.add_vertex('root', **value)

    p_id = [0]
    recursive_clustering_test_stop(original_data, original_data_classes, 'all', datatypes_used, joints, i_graph, p_id, debug=debug)

    # Assigning colours to the vertices (dict is pretty self-explanatory)
    color_dict = {'black': [0, 0.25], 'red':[0.25, 0.5], 'orange': [0.5, 0.75], 'green': [0.75, 1]}
    for i,vertex in enumerate(i_graph.vs):
        # vertex['label'] = i
        if vertex['ss'] == 1:
            vertex['color'] = 'blue'
        if vertex['best']:
            vertex['shape'] = 'rectangle'
        else:
            for k, v in color_dict.items():
                if v[0] <= max(vertex['c_together']) < v[1]:
                    vertex['color'] = k
                    print(f'{max(vertex["c_together"])}: color {k}')

    bbox = (1000, 1000)
    igraph.plot(i_graph, "output_graph_tree.svg", layout=i_graph.layout('tree'), vertex_size=5, bbox=bbox)
    igraph.plot(i_graph, "output_graph_large.svg", layout=i_graph.layout('large_graph'), vertex_size=5, bbox=bbox)
    igraph.plot(i_graph, "output_graph_circular.svg", layout=i_graph.layout('rt_circular'), vertex_size=5, bbox=bbox)
    i_graph.save("output.gml", format="gml")
    i_graph.save("output.gml", format="graphml")

    if debug:
        print(f'Iterations: {ITERATION_COUNTER}')

def recursive_clustering(original_data, data_to_keep, datatype_used, joints, i_graph, p_id, debug=False):
    """
        Performs a recursive clustering, as long as SS > THRESHOLD
        and sample_n > 2
    """

    # TODO: something without this (because there'll be a huge memory
    # consumption)

    global ITERATION_COUNTER

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
                if debug:
                    print(f"MAX_K_VALUE TOO LOW {max_k_value}")
                continue

            for k in range(2, max_k_value + 1):
                ITERATION_COUNTER += 1
                if debug:
                    print(f'Computing for k={k}, data = {data}, joint = {joint} (p_id = {p_id}, max_k_value = {max_k_value})')
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

                if debug:
                    print(f'SS: {s_score}')

                idx_c = []
                value = {}
                value['data'] = data
                value['joint'] = joint
                value['ss'] = s_score
                # We get the indexes of motion in every clusters
                for c in range(len(np.unique(res.labels_))):
                    idx_c.append(np.where(res.labels_ == c)[0].tolist())
                for i,rep in enumerate(idx_c):
                    value[f'c{i}'] = [x.name for i, x in enumerate(original_data) if i in idx_c[0]]

                i_graph.add_vertex(**value)
                i_graph.add_edge(p_id[-1], len(i_graph.vs) - 1)
                if debug:
                    print(f'Link {p_id[-1]} and {len(i_graph.vs) - 1}')



                 # If it's value > threshold:
                if s_score < SS_THRESHOLD:
                    if debug:
                        print(f'SS TOO LOW')
                    pass
                else:

                    p_id.append(len(i_graph.vs) - 1)

                    if debug:
                        print(f'SS OK')
                        print(f'new _pid: {p_id}')

                    # Recursive clustering on each cluster
                    for i,rep in enumerate(idx_c):
                        # If we have less than 3 samples in a cluster,
                        # no need to perfom clustering on it
                        if len(rep) < 3:
                            if debug:
                                print(f'EMPTY CLUSTER (n°{i})')
                        else:
                            if debug:
                                print(f'CLUSTER SUFFICIENTLY BIG (n°{i})')
                            names = [x.name for i, x in enumerate(original_data) if i in idx_c[0]]
                            recursive_clustering(original_data, names, datatype_used, joints, i_graph, p_id, debug)


                    p_id.pop()

    if debug:
        print(f'new p_id: {p_id}')

def recursive_clustering_test_stop(original_data, original_data_classes, data_to_keep, datatype_used, joints, i_graph, p_id, debug=False):
    """
        Performs a recursive clustering, as long as SS > THRESHOLD
        and sample_n > 2
    """

    # TODO: something without this (because there'll be a huge memory
    # consumption)

    global ITERATION_COUNTER

    sub_data = []
    sub_classes = []
    if data_to_keep == 'all':
        sub_data = original_data
        sub_classes = original_data_classes
    else:
        for motion in original_data:
            if motion.name in data_to_keep:
                sub_data.append(motion)
        for motion_class in original_data_classes:
            if motion_class[0] in data_to_keep:
                sub_classes.append(motion_class)

    for data in datatype_used:
        for joint in joints:

            max_k_value = 10

            if len(sub_data) <= max_k_value:
                max_k_value = len(sub_data) - 1

            # No need to compute a k-means for a sample number = 2
            if max_k_value <= 2:
                if debug:
                    print(f"MAX_K_VALUE TOO LOW {max_k_value}")
                continue

            res_k = []

            best_c_together = [0, 0]

            for k in range(2, max_k_value + 1):
                ITERATION_COUNTER += 1
                if debug:
                    print(f'Computing for k={k}, data = {data}, joint = {joint} (p_id = {p_id}, max_k_value = {max_k_value})')
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

                idx_c = []
                value = {}
                value['data'] = data
                value['joint'] = joint
                value['ss'] = s_score
                # We get the indexes of motion in every clusters
                for c in range(len(np.unique(res.labels_))):
                    idx_c.append(np.where(res.labels_ == c)[0].tolist())
                for i,rep in enumerate(idx_c):
                    value[f'c{i}'] = [x.name for i, x in enumerate(original_data) if i in idx_c[0]]

                clustering_res = {}
                clustering_res['k'] = k
                clustering_res['s_score'] = s_score
                clustering_res['idx_c'] = idx_c
                clustering_res['repartition'] = {}

                for i,rep in enumerate(idx_c):
                    clustering_res['repartition'][f'c{i}'] = [x.name for i, x in enumerate(original_data) if i in idx_c[0]]

                clustering_res['c_together'] = compute_clusters(res.labels_, sub_classes)
                res_k.append(clustering_res)

                if debug:
                    print(f'c_together: {max(clustering_res["c_together"])}')

                value['c_together'] = clustering_res['c_together']
                value['best'] = False

                if max(clustering_res['c_together']) > best_c_together[0]:
                    best_c_together[0] = max(clustering_res['c_together'])
                    best_c_together[1] = k - 2

                i_graph.add_vertex(**value)
                i_graph.add_edge(p_id[-1], len(i_graph.vs) - 1)
                if debug:
                    print(f'Link {p_id[-1]} and {len(i_graph.vs) - 1}')

            s_score_all = [x['s_score'] for x in res_k]
            to_keep = res_k[s_score_all.index(max(s_score_all))]

            i_graph.vs[len(i_graph.vs) - 1 - (max_k_value - 1 - best_c_together[1]) + 1]['best'] = True

            p_id.append(len(i_graph.vs) - 1 - (max_k_value - 1 - s_score_all.index(max(s_score_all))) + 1)

            if debug:
                print(f'Max c_together: {max(to_keep["c_together"])}')
                print(f'new p_id: {p_id}')

            if max(to_keep['c_together']) < 0.75:
                print(f'c_together too low, continuing...')
                # Recursive clustering on each cluster
                for i,rep in enumerate(to_keep['idx_c']):
                    # If we have less than 3 samples in a cluster,
                    # no need to perfom clustering on it
                    if len(rep) < 3:
                        if debug:
                            print(f'EMPTY CLUSTER (n°{i})')
                    else:
                        if debug:
                            print(f'CLUSTER SUFFICIENTLY BIG (n°{i})')
                        names = [x.name for i, x in enumerate(original_data) if i in idx_c[0]]
                        recursive_clustering_test_stop(original_data, original_data_classes, names, datatype_used, joints, i_graph, p_id, debug)


                p_id.pop()


    if debug:
        print(f'new p_id: {p_id}')


if __name__ == '__main__':
    #init(c_to_keep=0, debug=True)
    init_test_stop(debug=True)
