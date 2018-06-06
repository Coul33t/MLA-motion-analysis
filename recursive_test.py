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

SS_THRESHOLD = 0.5

ITERATION_COUNTER = 0

def init(debug=False):
    """
        Initialise the data, and launch the recursive clustering
    """
    global ITERATION_COUNTER

    original_data = []
    original_data = json_import(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/Speed/Throw_ball_less/', 'Leo')
    datatypes_used = [['BegMaxEndSpeedNorm'], ['BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz'], ['BegMaxEndSpeedNorm', 'BegMaxEndSpeedDirx', 'BegMaxEndSpeedDiry', 'BegMaxEndSpeedDirz']]
    datatypes_used = [['BegMaxEndSpeedNorm'], ['BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz']]
    joints = [['RightHand']]


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
    color_dict = {'black': [0, 0.25], 'red':[0.25, 0.5], 'orange': [0.5, 0.70], 'green': [0.70, 1]}
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




if __name__ == '__main__':
    init(debug=True)
