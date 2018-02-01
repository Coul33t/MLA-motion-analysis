from sklearn.cluster import KMeans

def kmeans_algo(data, algo='k-means++', k=2, verbose=False, extrem_verbose=False):

    if verbose:
        print("\n\n{}\n".format(algo))

    # About fit vs fit_predict: https://stackoverflow.com/questions/25012342/scikit-learns-k-means-what-does-the-predict-method-really-do
    res = KMeans(k, algo, n_init=20, max_iter=10000, verbose=extrem_verbose).fit(list(data))

    return res
