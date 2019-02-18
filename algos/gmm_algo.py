from sklearn.mixture import GaussianMixture

def gmm_algo(data, params):
    res = GaussianMixture(**params).fit(list(data))
    res.labels_ = res.predict(data)
    return res