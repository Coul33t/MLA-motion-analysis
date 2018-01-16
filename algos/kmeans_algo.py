from sklearn.cluster import KMeans

def kmeans_algo(data, algo='k-means++', verbose=False):

    if verbose:
        print("\n\n{}\n".format(algo))

    res = KMeans(2, algo, n_init=20, max_iter=1000).fit(list(data))

    return res
