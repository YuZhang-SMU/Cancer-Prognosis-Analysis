import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import mahotas
import os, time, random

data_path = '../KIRC_his_all' # File location of path after unified magnification and histogram matching
patient_list1 = os.listdir(data_path)
patient_list2 = [i[0:12] for i in patient_list1]

for m in range(len(patient_list1)):
    image_list = os.listdir(os.path.join(data_path, patient_list1[m]))
    print('patient %d' % m)
    for times in range(int(len(image_list))):
        image_path = os.path.join(data_path,patient_list1[m], image_list[times])
        image = mpimg.imread(image_path)
        image_uint8 = image.astype('uint8')
        feature = mahotas.features.pftas(image_uint8)
        feature = feature[np.newaxis,:]
        if times == 0:
            one_patient = feature
        else:
            one_patient = np.vstack((one_patient,feature))

    one_patient1 = np.mean(one_patient,axis=0)
    if m == 0:
        feature_matrix = one_patient1
    else:
        feature_matrix = np.vstack((feature_matrix,one_patient1))
