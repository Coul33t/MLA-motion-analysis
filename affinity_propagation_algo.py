from sklearn.cluster import AffinityPropagation

def affinity_propagation_algo(data, verbose=False):
    algo_accuracy = []

    # Which affinity to use
    algo = 'affinity'
        
    accuracy = []
    
    if verbose:
        print("\n\n{}\n".format(algo))            

    res = AffinityPropagation().fit(list(data))

    true_labels = [0,0,1,0,0,     0,0,0,0,1,      1,0,0,0,0,      0,1,1,0,1]

    diff = []
    for i,_ in enumerate(true_labels):
        diff.append(abs(true_labels[i]-res.labels_[i]))

    accuracy.append(max(diff.count(0)/len(diff), diff.count(1)/len(diff)))
    
    if verbose:
        print('Done.')

    algo_accuracy = [algo, accuracy]

    if verbose:
        print("Algorithm        : {}".format(algo))
        print("Accuracy         : {}".format(algo_accuracy[i][1]))
        print("Highest accuracy : {}".format(max(algo_accuracy[i][1])))
        print("Lowest accuracy  : {}".format(min(algo_accuracy[i][1])))
        print("Mean accuracy    : {}".format(sum(algo_accuracy[i][1])/len(algo_accuracy[i][1])))
        print("Median accuracy  : {}".format(median(algo_accuracy[i][1])))
        print('\n\n\n')

    return (algo_accuracy)