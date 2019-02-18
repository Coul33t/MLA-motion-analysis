from tools import file_name_gathering, natural_keys
from data_import import return_files
import os
import json
import pdb

folder_path = r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/Esteban'
folder_path = r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/BFC/SCG'
folder_path = r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/test_export_class/Leo'
#folder_path = r'C:/Users/quentin/Documents/These/Databases/Res/all/Right-handed'

def gather(metric_name='ss'):
    res_sup = []

    for root, subdirs, files in os.walk(folder_path):

        if files:

            for file in files:

                res = []

                if '.json' in file:

                    with open((root + '/' + file).replace('\\', '/')) as json_file:
                        try:
                            res = json.load(json_file)
                        except json.decoder.JSONDecodeError:
                            json_file.seek(0)

                        for subres in res:
                            # if subres['metrics']['ars'] >= 0.1:
                            #     res_sup.append(subres)
                            if metric_name in subres['metrics'].keys():
                                res_sup.append(subres)
    return res_sup

def print_metric(res, metric='ars'):
    for elem in res:
        print(f'{elem["metrics"][metric]}')

def main_1():
    res_sup = gather(metric_name='ars')
    print_metric(res_sup)

    pdb.set_trace()
    for elem in res_sup:
        print(f"{elem['data_used']} / {elem['joints_used']} / {elem['metrics']['ss']}")
    yo = []
    for elem in res_sup:
        yo.append(elem['metrics']['ars'])
    mean = sum(yo) / len(yo)
    pdb.set_trace()
    yo = []
    for elem in res_sup:
        if elem['metrics']['ss'] > 0.75:
            yo.append(elem)

    lst_values = {}
    for elem in yo:
        lst_values.setdefault(elem['data_used'], []).append(elem['metrics']['ss'])

    for k in lst_values.keys():
        print(f"{k}: {len(lst_values[k])}")
    pdb.set_trace()

def main_2():
    import networkx as nx

    g = nx.read_graphml('output.graphml')

    best_nodes = []

    for n in g.nodes(data=True):
        if n[1]['best'] == True:
            best_nodes.append(n[1])

    pdb.set_trace()

if __name__ == '__main__':
    main_2()
# datatypes = set()
# for elem in res_sup:
#     datatypes.add(elem['data_used'])

# joints = set()
# for elem in res_sup:
#     joints.add(elem['joints_used'])

# joints = list(joints)

# data_to_keep = ['BegMaxEndSpeedx, BegMaxEndSpeedy, BegMaxEndSpeedz',
#                 'BegMaxEndSpeedNorm, BegMaxEndSpeedDirx, BegMaxEndSpeedDiry, BegMaxEndSpeedDirz',
#                 'BegMaxEndSpeedNorm'
#                ]
# subres = []

# for elem in res_sup:
#     if elem['data_used'] in data_to_keep:
#         subres.append(elem)

# for d in data_to_keep:
#     for j in joints:
#         maxi = []
#         for elem in subres:
#             if (elem['data_used'] == d and elem['joints_used'] == j):
#                 if not maxi:
#                     maxi = elem
#                 else:
#                     if elem["metrics"]["ss"] > maxi["metrics"]["ss"]:
#                         maxi = elem
#         print(f'{maxi["data_used"]}\n{maxi["joints_used"]}\nk = {maxi["k"]}\n{maxi["metrics"]["ss"]}\n')
#         pdb.set_trace()

