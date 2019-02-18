from sklearn.cluster import MeanShift

def mean_shift_algo(data, params):
    res = MeanShift(**params).fit(list(data))

    return res
