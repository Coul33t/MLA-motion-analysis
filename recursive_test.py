import pdb

import numpy as np

import igraph
import networkx as nx

from data_import import json_import

from data_processing import (joint_selection,
                             data_selection)

from algos.kmeans_algo import kmeans_algo

from algos.metrics import (adjusted_rand_score_computing,
                           silhouette_score_computing,
                           compute_all_gt_metrics,
                           compute_all_clustering_metrics)

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

def compute_clusters(labels, gt, nb_c_labels):
    """
        This function compute the relative proportion of each class
        in each cluster.
        let say we have 100 samples of class +, and 100 samples of
        class -.
        c0 : [99+, 99-]
        c1 : [1+, 1-]

        the result will be
        c0 : [50%+, 50%-]
        c1 : [50%+, 50%-]
    """

    n_labels = labels
    n_gt = [x[1] for x in gt]

    if not isinstance(n_labels, np.ndarray):
        n_labels = np.array(n_labels)
    if not isinstance(n_gt, np.ndarray):
        n_gt = np.array(n_gt)

    clusters = []

    for i in range(nb_c_labels):
        clusters.append([])

    for i in range(len(n_gt)):
        clusters[n_labels[i]].append(n_gt[i])

    res = []
    res2 = []
    for cluster in clusters:
        tmp = []
        tmp2 = []
        for i in range(len(np.unique(n_gt))):
            tmp.append(cluster.count(i) / len(cluster))
            tmp2.append(cluster.count(i))
        res.append(tmp)
        res2.append(tmp2)

    return res, res2


def graph_analysis(graph):
    best_nodes = []
    c_together = []

    for node in graph.vs:
        if node['c_together_percentage'] and max(max(node['c_together_percentage'])) > 0.75:
            c_together.append(node)
        if node['best'] == True:
            best_nodes.append(node)

    print("BEST NODES:")
    for elem in best_nodes:
        print("\n--------------------------------")
        print(f"Data: {elem['data']}")
        print(f"k: {elem['k']}")
        print(f"Number of data points: {elem['nb_data']}")
        print(f"Joint: {elem['joint']}")
        print(f"Recall: {elem['recall']}")
        print(f"Precision: {elem['precision']}")
        print(f"F score: {elem['f_score']}")
        print(f"Average Silhouette Score: {elem['ss']}")
        print(f"Adujsted Rand Score: {elem['ars']}")
        print(f"Class repartition: {elem['c_together_percentage']}")
        print(f"Class repartition: {elem['c_together_numbers']}")
        for string in elem['res_string']:
            print(f"{string} ->")
        print(f"{elem['data']}, {elem['joint']}, {elem['k']}")

    # print("\n\n\nCLASS REPARTITION > 0.75:")
    # for elem in c_together:
    #     print("\n--------------------------------")
    #     print(f"Data: {elem['data']}")
    #     print(f"k: {elem['k']}")
    #     print(f"Number of data points: {elem['nb_data']}")
    #     print(f"Joint: {elem['joint']}")
    #     print(f"Recall: {elem['recall']}")
    #     print(f"Precision: {elem['precision']}")
    #     print(f"F score: {elem['f_score']}")
    #     print(f"Average Silhouette Score: {elem['ss']}")
    #     print(f"Adujsted Rand Score: {elem['ars']}")
    #     print(f"Class repartition: {elem['c_together_percentage']}")
    #     print(f"Class repartition: {elem['c_together_numbers']}")
    #     for string in elem['res_string']:
    #         print(f"{string} ->")
    #     print(f"{elem['data']}, {elem['joint']}, {elem['k']}")

def export_node(n):
    pdb.set_trace()

def graph_text_export(g, filename):
    pdb.set_trace()

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
    i_graph.save("output.graphml", format="graphml")

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
    datatypes_used = [['BegMaxEndSpeedNorm'], ['BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz'], ['BegMaxEndSpeedNorm', 'BegMaxEndSpeedDirx', 'BegMaxEndSpeedDiry', 'BegMaxEndSpeedDirz']]
    joints = [['RightHand']]

    original_data = original_data[:50]
    labels = dl.LEO_LABELS_2[:50]

    original_data_classes = motion_classes(original_data, labels)


    # +1 for the root
    i_graph = igraph.Graph()
    value = {}
    value['ss'] = 1
    value['datatypes'] = datatypes_used
    value['joints'] = joints
    value['datanames'] = [x.name for x in original_data]
    i_graph.add_vertex('root', **value)

    p_id = [0]
    res_string = []
    recursive_clustering_test_stop(original_data, original_data_classes, 'all', datatypes_used, joints, 0.9, i_graph, p_id, res_string, debug=debug)

    graph_analysis(i_graph)

    # Assigning colours to the vertices (dict is pretty self-explanatory)
    color_dict = {'black': [0, 0.25], 'red':[0.25, 0.5], 'orange': [0.5, 0.75], 'green': [0.75, 1]}
    for i,vertex in enumerate(i_graph.vs):
        # vertex['label'] = i
        if vertex['best']:
            vertex['shape'] = 'rectangle'

        if vertex['ss'] == 1:
            vertex['color'] = 'blue'
        else:
            for k, v in color_dict.items():
                if v[0] <= max(max(vertex['c_together_percentage'])) < v[1]:
                    vertex['color'] = k

    # bbox = (1000, 1000)
    # igraph.plot(i_graph, "output_graph_tree.svg", layout=i_graph.layout('tree'), vertex_size=5, bbox=bbox)
    # igraph.plot(i_graph, "output_graph_large.svg", layout=i_graph.layout('large_graph'), vertex_size=5, bbox=bbox)
    # igraph.plot(i_graph, "output_graph_circular.svg", layout=i_graph.layout('rt_circular'), vertex_size=5, bbox=bbox)
    # igraph.plot(i_graph, "output_graph_tree.png", layout=i_graph.layout('tree'), vertex_size=5, bbox=bbox)
    # igraph.plot(i_graph, "output_graph_large.png", layout=i_graph.layout('large_graph'), vertex_size=5, bbox=bbox)
    # igraph.plot(i_graph, "output_graph_circular.png", layout=i_graph.layout('rt_circular'), vertex_size=5, bbox=bbox)

    # i_graph.save("output.gml", format="gml")
    # i_graph.save("output.graphml", format="graphml")
    # graph_text_export(i_graph, 'output.txt')

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

                if k == 2:
                    value['ars'] = adjusted_rand_score_computing(res.labels_, sub_classes)
                    dic = compute_all_gt_metrics(res.labels_, [x[1] for x in sub_classes])
                    for key, val in dic.items():
                        if key not in value.keys():
                            value[key] = val


                dic = compute_all_clustering_metrics(features, res.labels_)
                for key, val in dic.items():
                    if key not in value.keys():
                        value[key] = val

                # We get the indexes of motion in every clusters
                for c in range(len(np.unique(res.labels_))):
                    idx_c.append(np.where(res.labels_ == c)[0].tolist())

                for i,rep in enumerate(idx_c):
                    value[f'c{i}'] = [x.name for j, x in enumerate(original_data) if j in idx_c[i]]

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

def recursive_clustering_test_stop(original_data, original_data_classes,
                                   data_to_keep, datatype_used, joints,
                                   rep_threshold, i_graph, p_id, res_string,
                                   debug=False):
    """
        Performs a recursive clustering, as long as ASS > THRESHOLD
        and sample_n > 2
    """

    # Debug value for iteration count
    global ITERATION_COUNTER

    # print(f"New clustering on {len(data_to_keep)} data points")
    # print(f"res_string: {res_string}")

    sub_data = []
    sub_classes = []

    # Keep all data
    if data_to_keep == 'all':
        sub_data = original_data
        sub_classes = original_data_classes

    # Choose the data (by name)
    else:
        for motion in original_data:
            if motion.name in data_to_keep:
                sub_data.append(motion)

        for motion_class in original_data_classes:
            if motion_class[0] in data_to_keep:
                sub_classes.append(motion_class)

    # Max value for the k-means (limit iteration number)
    max_k_value = 10

    # If there is less data left than the max k value,
    # set it equal to the number of data - 1
    if len(sub_data) <= max_k_value:
        max_k_value = len(sub_data) - 1

    if debug:
        print(f"MAX K VALUE {max_k_value}")

    # No need to compute a k-means for a sample number = 2
    if max_k_value <= 2:
        if debug:
            print(f"MAX_K_VALUE TOO LOW {max_k_value}")
        return

    # Doesn't seem to be used, to delete?
    nb_iter = 0

    # Stores the clustering results for all the passes (k -> 2-10, data, joint)
    res_k = []

    best_c_together = [0, 0]

    # C l u s t e r i n g
    for k in range(2, max_k_value + 1):
        # For each data type combination
        for data in datatype_used:
            # for each joint combination
            for joint in joints:

                ITERATION_COUNTER += 1
                nb_iter += 1

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
                # value will contain info to export to graph
                graph_node = {}
                graph_node['data'] = data
                graph_node['joint'] = joint
                graph_node['k'] = k
                graph_node['nb_data'] = len(features)
                graph_node['res_string'] = res_string

                # GT metrics
                if k == 2:
                    dic = compute_all_gt_metrics(res.labels_, [x[1] for x in sub_classes])
                    for key, val in dic.items():
                        if key not in graph_node:
                            graph_node[key] = val

                # Clustering (no labels) metrics
                dic = compute_all_clustering_metrics(features, res.labels_)
                for key, val in dic.items():
                    if key not in graph_node.keys():
                        graph_node[key] = val

                # We get the indexes of motion in every clusters
                for c in range(len(np.unique(res.labels_))):
                    idx_c.append(np.where(res.labels_ == c)[0].tolist())

                # Clusters composition by names
                for i,rep in enumerate(idx_c):
                    graph_node[f'c{i}'] = [x.name for j, x in enumerate(original_data) if j in idx_c[i]]

                # Clustering's general info + results will be stored in there
                clustering_res = {}
                clustering_res['k'] = k
                clustering_res['data'] = data
                clustering_res['joint'] = joint
                clustering_res['res_string'] = res_string
                clustering_res['nb_data'] = len(features)
                clustering_res['s_score'] = s_score
                clustering_res['idx_c'] = idx_c
                clustering_res['repartition'] = {}

                # Clusters composition by names
                for i,rep in enumerate(idx_c):
                    clustering_res['repartition'][f'c{i}'] = [x.name for i, x in enumerate(original_data) if i in idx_c[0]]

                # Clusters stats (percentage and number)
                clustering_res['c_together_percentage'], clustering_res['c_together_numbers'] = compute_clusters(res.labels_, sub_classes, len(np.unique(res.labels_)))

                res_k.append(clustering_res)

                if debug:
                    print(f'c_together_percentage: {max(clustering_res["c_together_percentage"])}')
                    print(f'ss: {clustering_res["s_score"]}')

                graph_node['c_together_percentage'] = clustering_res['c_together_percentage']
                graph_node['c_together_numbers'] = clustering_res['c_together_numbers']
                graph_node['best'] = False

                if max(max(clustering_res['c_together_percentage'])) > best_c_together[0]:
                    best_c_together[0] = max(max(clustering_res['c_together_percentage']))
                    best_c_together[1] = k - 2

                i_graph.add_vertex(**graph_node)

                # Not working at all
                i_graph.add_edge(p_id[-1], len(i_graph.vs) - 1)
                if debug:
                    print(f'Link {p_id[-1]} and {len(i_graph.vs) - 1}')

            # END JOINT COMBINATIONS
        # END DATATYPE COMBINATIONS
    # END CLUSTERING

    # Compute which clustering gave the best result (by ASS)
    s_score_all = [x['s_score'] for x in res_k]
    to_keep = res_k[s_score_all.index(max(s_score_all))]

    # Used to construct the graph (Idk if it's working or not atm)
    # UPDATE: it's not (broken af)
    i_graph.vs[len(i_graph.vs) - 1 - (max_k_value - 1 - best_c_together[1]) + 1]['best'] = True

    p_id.append(len(i_graph.vs) - 1 - (max_k_value - 1 - s_score_all.index(max(s_score_all))) + 1)

    # Which cluster should be recursively clustered
    # Current condition: one class is represented > 75%
    cluster_to_process = []

    for cluster_rep in to_keep['c_together_percentage']:
        if max(cluster_rep) < rep_threshold:
            cluster_to_process.append(1)
        else:
            cluster_to_process.append(0)

    if debug:
        print(f'Max c_together_percentage: {max(to_keep["c_together_percentage"])}')
        print(f'new p_id: {p_id}')

    ##############################################################################
    #TODO: do clustering for all sub_clusters that don't have the threshold value#
    ##############################################################################

    res_string.append(f"{to_keep['data']}, {to_keep['joint']}, {to_keep['k']}")

    # If there's at least one cluster to process
    if sum(cluster_to_process) > 0:
        print(f'c_together too low, continuing...')

        # For each cluster
        for i,rep in enumerate(to_keep['idx_c']):

            # If the cluster must not be processed, pass
            if cluster_to_process[i] == 0:
                if debug:
                    print(f'NO NEED TO PROCESS CLUSTER {i} (REP > 0.75)')
                continue

            # If we have less than 3 samples in a cluster,
            # no need to perfom clustering on it
            if len(rep) < 3:
                if debug:
                    print(f'EMPTY CLUSTER (n°{i})')
                continue
            else:
                if debug:
                    print(f'CLUSTER SUFFICIENTLY BIG (n°{i})')
                    print(f'(max ss = {to_keep["s_score"]})')
                    print(f'(all s_score: {s_score_all})')

                names = [x.name for i, x in enumerate(original_data) if i in idx_c[0]]
                recursive_clustering_test_stop(original_data, original_data_classes,
                                               names, datatype_used, joints, rep_threshold,
                                               i_graph, p_id, res_string, debug)

    p_id.pop()


    if debug:
        print(f'new p_id: {p_id}')


if __name__ == '__main__':
    #init(c_to_keep=0, debug=True)
    init_test_stop(debug=False)
