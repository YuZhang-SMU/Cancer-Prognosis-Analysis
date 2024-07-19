import torch
import os
import random
import pickle
import torchvision

def get_tile_map(dataset, if_load = True):
    path = f"/home/user/tuchao/project/project4/code/cla_connect/data/{dataset}.pkl"
    if if_load and os.path.exists(path):
        with open(path, "rb") as tf:
            tile_map = pickle.load(tf)

    else:
        data_dir = f'/home/data/tuchao/ms_diverse_size/{dataset}_ms/data/5X'
        tile_map = {} # key:slideID
        for i in os.listdir(data_dir): # make tile ID lists for all slides
            tiles = os.listdir(f'{data_dir}/{i}')
            random.shuffle(tiles)
            tile_map[i] = tiles

        with open(path, "wb") as tf:
            pickle.dump(tile_map,tf)
    return tile_map

def build_inst_multi(tile_map, slideID, class_label, inst_size, bag_size, args):

    tile_dir_5x = f'/home/data/tuchao/ms_diverse_size/{args.dataset}_ms/data/5X'
    tile_dir_10x = f'/home/data/tuchao/ms_diverse_size/{args.dataset}_ms/data/10X'
    tile_dir_20x = f'/home/data/tuchao/ms_diverse_size/{args.dataset}_ms/data/20X'
    inst_list = []
    for k, v in tile_map.items():
        if k[:12] == slideID:
            tile_i_list = v # list of tile ID for a slide
        else:
            continue
        inst_num = int(len(tile_i_list)/inst_size)

        for i in range(inst_num):
            inst_5x = []
            inst_10x = []
            inst_20x = []
            start = i*inst_size
            end = start + inst_size

            for j in range(start, end):
                tile_i = tile_i_list[j]
                inst_5x.append(f'{tile_dir_5x}/{k}/{tile_i}')
                inst_10x.append(f'{tile_dir_10x}/{k}/{tile_i}')
                inst_20x.append(f'{tile_dir_20x}/{k}/{tile_i}')
            inst_list.append([inst_5x, inst_10x, inst_20x, class_label, slideID])

    tile_num = len(inst_list) # the number of tiles for a instance
    if tile_num < bag_size or bag_size == -1:
        bag_size = inst_num
    return inst_list[:bag_size]

def data_load_single(data, device):
    tensor_list = [torchvision.io.decode_jpeg(torchvision.io.read_file(img_path), device=device).float().div_(255) for img_path in data[0]]
    input_tensor = torch.stack(tensor_list, dim=0)

    return input_tensor, torch.LongTensor([data[3]])

def data_load_double(data, device):
    tensor_list_x1 = [torchvision.io.decode_jpeg(torchvision.io.read_file(img_path), device=device).float().div_(255) for img_path in data[0]]
    tensor_list_x2 = [torchvision.io.decode_jpeg(torchvision.io.read_file(img_path), device=device).float().div_(255) for img_path in data[1]]
    input_tensor_x1 = torch.stack(tensor_list_x1, dim=0)
    input_tensor_x2 = torch.stack(tensor_list_x2, dim=0)

    return [input_tensor_x1, input_tensor_x2], torch.LongTensor([data[3]])

def data_load_multi(data, device):
    tensor_list_x1 = [torchvision.io.decode_jpeg(torchvision.io.read_file(img_path), device=device).float().div_(255) for img_path in data[0]]
    tensor_list_x2 = [torchvision.io.decode_jpeg(torchvision.io.read_file(img_path), device=device).float().div_(255) for img_path in data[1]]
    tensor_list_x3 = [torchvision.io.decode_jpeg(torchvision.io.read_file(img_path), device=device).float().div_(255) for img_path in data[2]]
    input_tensor_x1 = torch.stack(tensor_list_x1, dim=0)
    input_tensor_x2 = torch.stack(tensor_list_x2, dim=0)
    input_tensor_x3 = torch.stack(tensor_list_x3, dim=0)

    return [input_tensor_x1, input_tensor_x2, input_tensor_x3], torch.LongTensor([data[3]])