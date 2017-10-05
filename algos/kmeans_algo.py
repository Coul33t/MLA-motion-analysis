from sklearn.cluster import KMeans

def kmeans_algo(data, verbose=False):

    algo_accuracy = []

    # Kmeans initialization type
    init = ['kmeans++', 'random', 'ndarray']

    for algo in init:
        
        accuracy = []
        
        if verbose:
            print("\n\n{}\n".format(algo))

        for i in range(10):
            
            if verbose: 
                print('Iteration {} ... '.format(i+1), end=' ')

            res = KMeans(2, 'k-means++', n_init=20, max_iter=1000).fit(list(data))

            true_labels = [0,0,1,0,0,     0,0,0,0,1,      1,0,0,0,0,      0,1,1,0,1]

            diff = []
            for i,_ in enumerate(true_labels):
                diff.append(abs(true_labels[i]-res.labels_[i]))

            accuracy.append(max(diff.count(0)/len(diff), diff.count(1)/len(diff)))
            
            if verbose:
                print('Done.')

        algo_accuracy.append([algo, accuracy])

    if verbose:
        for i,algo in enumerate(init):
            print("Algorithm        : {}".format(algo))
            print("Accuracy         : {}".format(algo_accuracy[i][1]))
            print("Highest accuracy : {}".format(max(algo_accuracy[i][1])))
            print("Lowest accuracy  : {}".format(min(algo_accuracy[i][1])))
            print("Mean accuracy    : {}".format(sum(algo_accuracy[i][1])/len(algo_accuracy[i][1])))
            print("Median accuracy  : {}".format(median(algo_accuracy[i][1])))
            print('\n\n\n')

    return ([algo_accuracy[i] for i in range(len(init))])
