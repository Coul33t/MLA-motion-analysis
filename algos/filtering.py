def mean_filter(data, window_size = 5):
    """
        Apply a mean filter on a list of data
    """
    if window_size % 2 == 0:
        print('Error; window_size must be odd.')
        return None

    if window_size > len(data):
        print('Error: window_size must be inferior to the data\'s length.')
        return None

    new_data = [0 for x in range(len(data))]

    for i, sample in enumerate(data):
        if i < int(window_size/2):

        new_data[i] = 