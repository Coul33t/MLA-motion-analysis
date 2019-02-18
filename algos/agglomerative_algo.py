from sklearn.cluster import AgglomerativeClustering

def agglomerative_algo(data, params):
    res = AgglomerativeClustering(**params).fit(data)

    return res
