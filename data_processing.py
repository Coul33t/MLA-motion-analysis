import OrderedDict

def delta_computing(data_dic):
    delta_dict = []

    for dic in data_dic:
        new_dic = OrderedDict()
        
        for key in dic.keys():
            delta_values = []

            for i,values in enumerate(dic[key][:-1]):
                delta_values.append(dic[key][i+1] - dic[key][i])

            new_dic[key] = delta_values#[min(delta_values), max(delta_values)]

        delta_dict.append(new_dic)

    return delta_dict


def mean_speed(data_dic):
    """
        Computes the global mean speed of a motion.
    """
    mean_dict = []

    for dic in data_dic:
        new_dic = OrderedDict()
        for key in dic.keys():
            new_dic[key] = sum(dic.get(key))/len(dic.get(key))
        mean_dict.append(new_dic)
    
    return mean_dict