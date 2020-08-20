import json
import numpy as np
import os
data_path='./v3/result/'
dirs = os.listdir(data_path)
for file in dirs:
    with open(data_path+file, 'r') as fin:
        data = json.load(fin)
    data_motion = np.array(data['skeletons'])
    center=data['center']
    for i in range(2,data_motion.shape[0]-2):
        if(np.sqrt(np.sum(np.square(data_motion[i] - data_motion[i+1])))>16):
            data_motion[i]=(data_motion[i]+data_motion[i+1])/2

    data = {"length": data['length'], "skeletons": data_motion.tolist(),"center":center}
    with open('./v3/interpolation_result/'+file, 'w') as file_object:
        json.dump(data, file_object)