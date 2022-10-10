#!/usr/bin/env python
# -*- coding:utf-8 -*-
import openslide
import numpy as np
import time
import scipy.io as scio
import os
import cv2
from openslide.deepzoom import DeepZoomGenerator
import warnings
import math
from skimage.exposure import match_histograms
from tqdm import tqdm
from PIL import Image

def extract_roi(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s2 = cv2.blur(s, (5, 5))
    _, saturation_thresholded_mask = cv2.threshold(s2, 15, 255, cv2.THRESH_BINARY)
    roi = saturation_thresholded_mask / 255
    return roi

def get_patch(path):
    object_path = path + '/svs'
    path = path + '/tiles'
    loc = path + '/location'
    if not os.path.exists(loc):
        os.makedirs(loc)
      
    fild_id_downloaded = os.listdir(object_path)
    num = fild_id_downloaded.__len__()
    print(num)

    for n in range(num):
        print('total number:', n + 1)
        file_name = fild_id_downloaded[n]
        print(file_name)
        
        spath = [f'{path}/{m}/{file_name[:23]}/' for m in ['5x','10x','20x']]
        for p in spath:
            if not os.path.exists(p): os.makedirs(p)
        
        slide = openslide.OpenSlide(os.path.join(object_path, file_name))
        
        data_128 = DeepZoomGenerator(slide, tile_size=128, overlap=0, limit_bounds=False)
        data_256 = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
        data_512 = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)
            
        [w, h] = slide.level_dimensions[0]
        level = data_128.level_count-1
        power = slide.properties['openslide.objective-power']
        
        if power not in ['20', '40']: continue
        
        if power == '40':
            w = math.ceil(w/2)
            h = math.ceil(h/2)
            level -= 1
            
        print(w, h)

        num_h = int(h / 512)
        num_w = int(w / 512)

        location = np.zeros((num_h, num_w))
        seq = 1

        for i in tqdm(range(num_h)):
            for j in range(num_w):
                img = [np.array(data_128.get_tile(level-2, (j, i)))]
                roi = extract_roi(img[0])
                rate = np.sum(roi) / (128 * 128)

                if rate > 0.3:
                    img.append(np.array(data_256.get_tile(level-1, (j, i))))
                    img.append(np.array(data_512.get_tile(level, (j, i))))
                    
                    for k in range(len(img)):
                        matched = match_histograms(img[k], ref, multichannel=True)
                        matched = Image.fromarray(matched)
                        matched.save('%s%d%s' % (spath[k], seq, '.jpg'))                 
                    
                    location[i, j] = 1
                    seq = seq + 1

        dataNew = loc + f'/{file_name[:23]}.mat'
        scio.savemat(dataNew, {'location': location})

    return 

if __name__ == '__main__':
    warnings.filterwarnings('error')
    time_start = time.time()

    ref = cv2.imread('./preprocessing/reference.tiff')
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    dataset = 'LIHC'
    main_path = f'./data/{dataset}'

    patches = get_patch(main_path)
        
    time_end = time.time()
    print('totally cost', time_end - time_start)
