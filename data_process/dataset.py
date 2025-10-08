import os

import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import re
import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
from scipy.io import loadmat
warnings.filterwarnings("ignore")

from tqdm import *
from sklearn.cluster import DBSCAN, mean_shift


SpectrumNetDatasetPath = ''

                    
def get_index(dataset_path):
    index_path = os.path.join(dataset_path,'png')
    seq = []
    for root, dirs, files in os.walk(index_path):
        for f in files:
            if 'z00' in f:
                seq.append(os.path.join(root, f))
    fo = open(os.path.join(dataset_path,"index.txt"), "w")
    fo.write('\n'.join(sorted(seq)))
    fo.close()

def get_info(dataset_path, file_path):
    file_name = file_path.split('/')[-1]

    frequencies_list = [15, 150 , 170, 350, 2200] # 单位10MHZ

    T, C, _, __, f = re.match(r"T(\d+)C(\d+)D(\d+)_n(\d+)_f(\d+)",file_name).groups()


    radiomap = np.stack([cv2.imread(file_path,0),cv2.imread(file_path.replace('z00','z01'),0),cv2.imread(file_path.replace('z00','z02'),0)])

    npz_path = os.path.join(dataset_path,'npz',file_name[:14]+'_bdtr.npz')
    mat = np.load(npz_path)
    building, terrain = mat['inBldg_zyx'], mat['terrain_yx']

    frequencies = frequencies_list[int(f)]
    env = int(T)

    return radiomap, building, terrain, frequencies, env


def point_cloud_sample(radiomap, building, sparse_num, point_cloud_height):
    HEIGHT_LIST = [0.15, 3, 20] # 单位十米
    if point_cloud_height < 3:
        p = 1-building[point_cloud_height].reshape(-1)
        p = p/np.sum(p)

        random_point = np.random.choice(128*128, sparse_num, replace=False, p = p)
        random_point_x = random_point//128
        random_point_y = random_point%128

        radio_ch = radiomap[point_cloud_height,random_point_x,random_point_y]
        height_ch = np.ones_like(radio_ch)*HEIGHT_LIST[point_cloud_height]

        point_cloud = np.stack([random_point_x,random_point_y,height_ch,radio_ch], axis=1)
    else:
        assert 0, "point_cloud_height输错了吧"
    
    return point_cloud

def mixed_point_cloud_sample(radiomap, building, sparse_num):
    HEIGHT_LIST = [0.15, 3, 20]
    point_cloud_list = []
    for point_cloud_height in range(3):
        p = 1-building[point_cloud_height].reshape(-1)
        p = p/np.sum(p)

        random_point = np.random.choice(128*128, sparse_num, replace=False, p = p)
        random_point_x = random_point//128
        random_point_y = random_point%128

        radio_ch = radiomap[point_cloud_height,random_point_x,random_point_y]
        height_ch = np.ones_like(radio_ch)*HEIGHT_LIST[point_cloud_height]

        point_cloud = np.stack([random_point_x,random_point_y,height_ch,radio_ch], axis=1)
        point_cloud = torch.tensor(point_cloud).to(torch.float32)
        point_cloud_list.append(point_cloud)
    
    point_cloud_pool = torch.cat(point_cloud_list,dim=0)

    point_cloud_list.append(point_cloud_pool[np.random.choice(point_cloud_pool.shape[0], sparse_num, replace=False)])
    
    return point_cloud_list


def mixed_sample(radiomap, building, sparse_num):

    sampled_map = np.full_like(radiomap, -1.0)
    
    free_indices = np.argwhere(building == 0)
    available_points = len(free_indices)

    selected_indices = np.random.choice(
        available_points, size=sparse_num, replace=False
    )

    selected_coords = free_indices[selected_indices].T
    sampled_map[selected_coords[0], selected_coords[1], selected_coords[2]] = \
        radiomap[selected_coords[0], selected_coords[1], selected_coords[2]]

        
    return sampled_map

def simple_sample(radiomap, building, sparse_num):
    sampled_map = np.full_like(radiomap, -1.0)

    for channel in range(3):
        free_indices = np.argwhere(building[channel] == 0)
        available_points = len(free_indices)

        selected_indices = np.random.choice(
            available_points, size=sparse_num, replace=False
        )

        for idx in selected_indices:
            y, x = free_indices[idx]
            sampled_map[channel, y, x] = radiomap[channel, y, x]

    return sampled_map


class SpectrumNetDataset(Dataset):
    def __init__(self, dataset_path, indices, sparse_num, point_cloud_height):
        self.dataset_path = dataset_path
        self.indices = indices
        self.point_cloud_height = point_cloud_height

        if not os.path.exists(os.path.join(dataset_path,"index.txt")):
            get_index(dataset_path)

        with open(os.path.join(dataset_path,"index.txt"),'r') as f:
            self.radio_maps = str(f.read()).split('\n')
        self.sparse_num = sparse_num


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        
        radiomap, building, terrain, frequencies, env = get_info(self.dataset_path, self.radio_maps[index])

        radiomap = radiomap*2/255 - 1 # Diffusion love (-1,1) normalize than (0,1) normalize because of normal noise
        terrain = (terrain + 2)/190

        if self.point_cloud_height != "mix":
            point_cloud = point_cloud_sample(radiomap, building, self.sparse_num, self.point_cloud_height)
            point_cloud = torch.tensor(point_cloud).to(torch.float32)
            sampled_map = simple_sample(radiomap, building, self.sparse_num)
        else:
            point_cloud = mixed_point_cloud_sample(radiomap, building, self.sparse_num)[-1]
            sampled_map = mixed_sample(radiomap, building, self.sparse_num)
       
        return torch.tensor(point_cloud).to(torch.float32), torch.tensor(sampled_map).view(3,128,128).to(torch.float32), torch.tensor(building).view(3,128,128).to(torch.float32), torch.tensor(terrain).view(1,128,128).to(torch.float32),  torch.tensor(frequencies).to(torch.float32), torch.tensor(env).to(torch.float32), torch.tensor(radiomap).view(3,128,128).to(torch.float32)
        


def random_sample(dense_image,num = 500, CHANNEL_SIZE = 5):
    random_point = np.random.choice(256*256, num, replace=False)
    random_point_x = random_point//256
    random_point_y = random_point%256

    sample_mat = np.zeros((CHANNEL_SIZE,256,256))
    sample_mat[:,random_point_x,random_point_y] = dense_image[:,random_point_x,random_point_y]

    return sample_mat


def get_transmitter_pos_GT(dataset_path):
    if not os.path.exists(os.path.join(dataset_path,"index.txt")):
        assert 0, 'you need get index first'

    with open(os.path.join(dataset_path,"index.txt"),'r') as f:
        file_list = str(f.read()).split('\n')

    seq = []
    for file_path in tqdm(file_list):
        radiomap = cv2.imread(file_path,0)
        X = np.argwhere(radiomap>np.max(radiomap)*0.8)

        weighted_centers, _ = mean_shift(X, bandwidth=10)
        seq.append(' '.join(map(lambda a: f"{int(np.around(a[0]))},{int(np.around(a[1]))}", weighted_centers)))

    fo = open(os.path.join(dataset_path,"transmitter_pos.txt"), "w")
    fo.write('\n'.join(seq))
    fo.close()

class TPNetDataset(Dataset):
    def __init__(self, dataset_path, indices, sparse_num, point_cloud_height):
        self.dataset_path = dataset_path
        self.indices = indices
        self.point_cloud_height = point_cloud_height

        if not os.path.exists(os.path.join(dataset_path,"index.txt")):
            get_index(dataset_path)
        if not os.path.exists(os.path.join(dataset_path,"transmitter_pos.txt")):
            get_transmitter_pos_GT(dataset_path)

        with open(os.path.join(dataset_path,"index.txt"),'r') as f:
            self.radio_maps = str(f.read()).split('\n')
        with open(os.path.join(dataset_path,"transmitter_pos.txt"),'r') as f:
            self.transmitter_pos = str(f.read()).split('\n')
        self.sparse_num = sparse_num


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        
        radiomap, building, terrain, frequencies, env = get_info(self.dataset_path, self.radio_maps[index])

        radiomap = radiomap*2/255 - 1 # Diffusion love (-1,1) normalize than (0,1) normalize because of normal noise
        terrain = (terrain + 2)/190

        point_cloud = point_cloud_sample(radiomap, building, self.sparse_num, self.point_cloud_height)

        transmitter_pos_mat = torch.zeros((128,128))

        for p in self.transmitter_pos[index].split(' '):
            p_x,p_y = p.split(',')
            transmitter_pos_mat[int(p_x),int(p_y)] = 1
       
        return torch.tensor(point_cloud).to(torch.float32), torch.tensor(building).view(3,128,128).to(torch.float32), torch.tensor(terrain).view(1,128,128).to(torch.float32),  torch.tensor(frequencies).to(torch.float32), torch.tensor(env).to(torch.float32), torch.tensor(transmitter_pos_mat).view(1,128,128).to(torch.float32)
        


def get_dataloader(type,
                   batch_size,
                   point_cloud_height,
                   sparse_num,
                   channel_num = "all",
                   Geographic_type = "all",
                   dist_train=False,
                   num_workers=20,
                   prefetch_factor=10):
    
    train_percent = 0.8

    print(f"sparse_num = {sparse_num}")
    print(f"point_cloud_height= {point_cloud_height}")
    print(f"channel_num = {channel_num}")
    print(f"Geographic_type = {Geographic_type}")

    dataset_path = SpectrumNetDatasetPath
    if not os.path.exists(os.path.join(dataset_path,"index.txt")):
        get_index(dataset_path)

    with open(os.path.join(dataset_path,"index.txt"),'r') as f:
        radio_maps = str(f.read()).split('\n')

    all_indices = {}
    for n, i in enumerate(radio_maps):
        if channel_num == 'all' or channel_num in i:
            label = i.split('/')[-2]
            if label not in all_indices:
                all_indices[label] = [n]
            else:
                all_indices[label].append(n)

    Geographic_items = {'all': all_indices.keys(), 
                        'urban': ['05.Suburban', '06.DenseUrban', '07.Rural', '08.OrdinaryUrban'],
                        'water': ['03.Ocean', '04.Lake'],
                        'nature': ['01.Grassland','02.Island','09.Desert','10.Mountainous','11.Forest']}
    
    train_indices = []
    test_indices = []

    if Geographic_type in all_indices.keys():
        type_data_size = len(all_indices[Geographic_type])

        train_indices += all_indices[Geographic_type][::2]
        train_indices += all_indices[Geographic_type][1::2][:int(type_data_size*(train_percent-0.5))]
        test_indices += all_indices[Geographic_type][1::2][int(type_data_size*(train_percent-0.5)):]

    elif Geographic_type in Geographic_items:
        for sub_geographic_type in Geographic_items[Geographic_type]:
            type_data_size = len(all_indices[sub_geographic_type])

            train_indices += all_indices[sub_geographic_type][::2]
            train_indices += all_indices[sub_geographic_type][1::2][:int(type_data_size*(train_percent-0.5))]
            test_indices += all_indices[sub_geographic_type][1::2][int(type_data_size*(train_percent-0.5)):]
    else:
        assert 0, "Geographic Type Error"

    print(f"data size = {len(train_indices) + len(test_indices)}")

    if type == 'SpectrumNet':
        dataset = SpectrumNetDataset(dataset_path=SpectrumNetDatasetPath, indices=train_indices, sparse_num=sparse_num, point_cloud_height=point_cloud_height)
    
    elif type == 'SpectrumNet_test':
        dataset = SpectrumNetDataset(dataset_path=SpectrumNetDatasetPath, indices=test_indices, sparse_num=sparse_num, point_cloud_height=point_cloud_height)
    else:
        assert 0, "Using Unknown Dataset"
        

    if dist_train:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=num_workers,
                                prefetch_factor=prefetch_factor)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                prefetch_factor=prefetch_factor)
        return dataloader



def getdata(n, sparse_num, point_cloud_height = 'mix'):
    dataset_path = SpectrumNetDatasetPath
    with open(os.path.join(dataset_path,"index.txt"),'r') as f:
        radio_maps = str(f.read()).split('\n')

    print(f"data path:{os.path.join(dataset_path, radio_maps[n])}")

    radiomap, building, terrain, frequencies, env = get_info(dataset_path, radio_maps[n])

    radiomap = radiomap*2/255 - 1 # Diffusion love (-1,1) normalize than (0,1) normalize because of normal noise
    terrain = (terrain + 2)/190

    if point_cloud_height != "mix":
        point_cloud = point_cloud_sample(radiomap, building, sparse_num, point_cloud_height)
        point_cloud = torch.tensor(point_cloud).to(torch.float32)
    else:
        sampled_map = mixed_sample(radiomap, building, sparse_num)
    
    return torch.tensor(sampled_map).view(1,3,128,128).to(torch.float32), torch.tensor(building).view(1,3,128,128).to(torch.float32), torch.tensor(terrain).view(1,1,128,128).to(torch.float32),  torch.tensor(frequencies).to(torch.float32), torch.tensor(env).to(torch.float32), torch.tensor(radiomap).view(1,3,128,128).to(torch.float32)
    
