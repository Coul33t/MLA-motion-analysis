def ml(motion_type="gimbal"):
    # DATA GATHERING
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data_lin = []
    data_ang = []
    
    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            print("Appending {}".format(name))
            data_lin.append(data_gathering_list(folder+'\\'+name+'\\lin'))

    # (Bad) NORMALIZATION
    min_size = min(map(len, data_lin))

    for i,elem in enumerate(data_lin):
        data_lin[i] = data_lin[i][10:min_size]

    #for name in subdirectories:
        #data_ang.append(data_gathering_list(folder+'\\'+name+'\\ang'))
        
    ## (Bad) NORMALIZATION
    #min_size = min(map(len, data_ang))

    #for i,elem in enumerate(data_ang):
        #data_ang[i] = data_ang[i][10:min_size]
    
    # for mo in data_lin:
    #     delta_computing(mo)

    kmeans_algo(data_lin)
    #kmeans_algo(data_ang)





def test_normalization(motion_type="gimbal"):
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data_lin = []
    data_lin_2 = []

    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            print("Appending {}".format(name))
            data_lin.append(data_gathering_dict(folder+'\\'+name+'\\lin'))
    
    # data_lin : list of dict (1 for each motion)
    # data_lin[0] : dict with joints keys (1 for each joint)
    # data_lin[0]['Hips'] : values of Hips data for the 1 motion (as much as frame number)
    

    data = []
    mean_len = []

    for dic in data_lin:
        mean_len.append(len(dic.get('Hips')))
    
    mean_len = int(sum(mean_len)/len(mean_len))

    # 10 because artefact in the 10th first frames (approximately)
    # mean_len/4 to avoid

    # (Bad) NORMALIZATION
    #min_size = min(map(len, data_lin))

    for i,_ in enumerate(data_lin):
        data_lin[i] = data_lin[i][mean_len/4:mean_len + (mean_len / 4)]

    mean_speed_data = mean_speed(data_lin)

    delta_speed_data = delta_computing(data_lin)

    new_data = []

    for i, _ in enumerate(mean_speed_data):
        tmp_list = []
        for key in data_lin[0].keys():
            tmp_list.append(mean_speed_data[i][key])
            tmp_list.extend(delta_speed_data[i][key])
        new_data.append(tmp_list)





def test_dtw(motion_type="gimbal"):
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    data_lin = []

    subdirectories = os.listdir(folder)
    subdirectories.sort(key=natural_keys)

    for name in subdirectories:
        if motion_type in name:
            print("Appending {}".format(name))
            data_lin.append(flatten_list(motion_dict_to_list(data_gathering_dict(folder+'\\'+name+'\\lin'))))
    
    # data_lin : list of dict (1 for each motion)
    # data_lin[0] : dict with joints keys (1 for each joint)
    # data_lin[0]['Hips'] : values of Hips data for the 1 motion (as much as frame number)
    
    pdb.set_trace()