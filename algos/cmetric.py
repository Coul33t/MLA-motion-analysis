class CMetric:
    def __init__(self):
        # List of dictionnaries
        self.c_composition = []


    def clusters_composition_name(labels, sample_nb, original_names, verbose=False):
    """
        For each k in labels, this function returns the clusters' composition (filenames).
    """

        c_composition = {}

        # Get a list of clusters number
        cluster_nb = set(labels)

        # Used to strip off the " Char00 " from the filename
        regex = re.compile(r'^\w*_\d+')

        stripped_names = np.asarray([regex.search(s).group() for s in original_names])

        # For each cluster
        for c in cluster_nb:
            c_composition['c' + str(c)] = stripped_names[np.where(labels == c)].tolist()

            if verbose:
                print("\nMotions in c{}: {}".format(c, stripped_names[np.where(labels == c)]))

        self.c_composition.append(c_composition)
        return c_composition