from sklearn.cluster import KMeans

def kmeans_algo(data, algo='k-means++', verbose=False):

    algo_accuracy = []

    # Kmeans initialization type:
    # 'k-means++', 'random', 'ndarray'
    
    if verbose:
        print("\n\n{}\n".format(algo))

    res = KMeans(2, algo, n_init=20, max_iter=1000).fit(list(data))

    return res
            
    #         if verbose:
    #             print('Done.')

    #     algo_accuracy.append([algo, accuracy])

    # if verbose:
    #     for i,algo in enumerate(init):
    #         print("Algorithm        : {}".format(algo))
    #         print("Accuracy         : {}".format(algo_accuracy[i][1]))
    #         print("Highest accuracy : {}".format(max(algo_accuracy[i][1])))
    #         print("Lowest accuracy  : {}".format(min(algo_accuracy[i][1])))
    #         print("Mean accuracy    : {}".format(sum(algo_accuracy[i][1])/len(algo_accuracy[i][1])))
    #         print("Median accuracy  : {}".format(median(algo_accuracy[i][1])))
    #         print('\n\n\n')

    # return ([algo_accuracy[i] for i in range(len(init))])
