from sklearn.cluster import DBSCAN

def dbscan_algo(data, params):
    res = DBSCAN(**params).fit(data)

    return res