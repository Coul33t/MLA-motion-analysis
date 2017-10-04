import pdb

import numpy as np

import sklearn.cluster as cl

#TODO: use pipes for communication
def main():
    data = [[0,0], [1,1], [0,1], [1,0], [0.2,1], [1,0.4]]

    # params:
    # n_clusters(default: 8)
    #
    # init (default: 'kmeans++', 'random', 'ndarray'): method of initialization
    #
    # n_nit (default: 10): number of run with different centroid seeds
    #
    # max_iter(default: 300)
    #
    # tol (default : 1e-4): tolerance to inertia to declare convergence
    #
    # precompute_distances (default: auto, True, False): faster but takes more memory
    #
    # verbose (default: 0)
    #
    # random_state (default: None): The generator used to initialize the centers. If an 
    #               integer is given, it fixes the seed. Defaults to the 
    #               global numpy random number generator.
    #
    # copy_x (default: True): When pre-computing distances it is more numerically accurate to
    #         center the data first. If copy_x is True, then the original data 
    #         is not modified. If False, the original data is modified, and 
    #         put back before the function returns, but small numerical 
    #         differences may be introduced by subtracting and then adding 
    #         the data mean.
    #
    # n_jobs (default: 1): number of jobs used to compute
    #
    # algorithm (default: 'auto', 'full', 'elkan'): K-means algorithm to use. The 
    #       classical EM-style algorithm is “full”. The “elkan” variation is more efficient by using the triangle inequality, but currently doesn’t support sparse data. “auto” chooses “elkan” for dense data and “full” for sparse data.
    res = cl.KMeans(4, 'k-means++', n_init=10, max_iter=1000).fit(data)
    pdb.set_trace()




if __name__ == '__main__':
    main()