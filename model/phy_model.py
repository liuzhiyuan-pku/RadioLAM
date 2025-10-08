import torch
import torch.nn as nn
import torchvision.models as models
import math, os

from sklearn.cluster import DBSCAN, mean_shift
import matplotlib.pyplot as plt

import cv2
import numpy as np
from tqdm import *

from scipy.optimize import least_squares
from functools import partial

import torch
import time

SpectrumNetDatasetPath = ''
SHOW_DATA_PATH = ""

DELTA_HEIGHT_LIST = [0, 2.85, 19.85] # 10m N-0.15


def get_transmitter_pos_GT(dataset_path):
    if not os.path.exists(os.path.join(dataset_path,"index.txt")):
        assert 0, "Get dataset index first"

    with open(os.path.join(dataset_path,"index.txt"),'r') as f:
        file_list = str(f.read()).split('\n')

    seq = []
    for file_path in tqdm(file_list):
        radiomap = cv2.imread(file_path,0)
        X = np.argwhere(radiomap>np.max(radiomap)*0.8)
        centers, _ = mean_shift(X, bandwidth=10)
        seq.append(' '.join(map(lambda a: f"{int(np.around(a[0]))},{int(np.around(a[1]))}", centers)))

    fo = open(os.path.join(dataset_path,"transmitter_pos.txt"), "w")
    fo.write('\n'.join(seq))
    fo.close()


class TransmitterPosNet(nn.Module):
    def __init__(self):
        super(TransmitterPosNet, self).__init__()
        self.fc = nn.Linear(256, 64 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64
        self.deconv4 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)  # 64x64 -> 128x128
        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 8, 8)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.tanh(self.deconv4(x))
        return x


def func(RSS,d,i):
    return np.asarray([RSS_v-np.sum(i/(d_v**2)) for RSS_v, d_v in zip(RSS,d)])

def get_transmitter_info(transmitter_pos, point_cloud):
    Pr_dBm = point_cloud[:,3]
    Pr_W = 10**(Pr_dBm)

    d = []
    for pos in transmitter_pos:
        d_i = np.linalg.norm(point_cloud[:,:3] - np.concatenate([pos,[0.15]]), axis=1)
        d.append(d_i)

    func_p = partial(func, Pr_W, np.stack(d,axis=1))

    init_val = np.ones(transmitter_pos.shape[0])*10
    try:
        root = least_squares(func_p, init_val, bounds=(0,np.inf)).x
    except Exception as e:
        print("!!!least_squares ERROR, use default value!!!")
        print(e)
        root = np.ones(transmitter_pos.shape[0])*10
    return 2, root



def func_fast(i, RSS, d_squared):
    return RSS - np.sum(i / d_squared, axis=1)

def jac_fast(i, RSS, d_squared):
    return -1 / d_squared

def get_transmitter_info_fast(transmitter_pos, point_cloud):
    Pr_dBm = point_cloud[:, 3]
    Pr_W = 10 ** Pr_dBm

    transmitters_3d = np.hstack([transmitter_pos, 0.15 * np.ones((transmitter_pos.shape[0], 1))])
    diffs = point_cloud[:, :3][:, np.newaxis, :] - transmitters_3d
    d = np.linalg.norm(diffs, axis=2)
    d_squared = d ** 2

    func_p = partial(func_fast, RSS=Pr_W, d_squared=d_squared)
    jac_p = partial(jac_fast, RSS=Pr_W, d_squared=d_squared)
    
    init_val = np.ones(transmitter_pos.shape[0]) * 10
    try:
        result = least_squares(
            func_p, 
            init_val, 
            jac=jac_p, 
            bounds=(0, np.inf), 
            method='trf',
            xtol=1e-6,
            ftol=1e-6,
            max_nfev=200
        )
        root = result.x
    except Exception as e:
        print("!!! least_squares ERROR，use default value !!!")
        print(e)
        root = np.ones(transmitter_pos.shape[0]) * 10
    
    return 2, root




def pos2radiomap(transmitter_pos, p_1, building, PLE, size=128, LogPathLoss = False):
    B = 3
    S = transmitter_pos.shape[0]  
    m, n = size, size
    noise_std = 0.05

    transmitter_pos = torch.tensor(transmitter_pos)

    x = torch.arange(m).view( -1, 1).repeat(S, 1, n)
    y = torch.arange(n).view( 1, -1).repeat(S, m, 1)


    pos_x = transmitter_pos[:,0].view(S, 1, 1)
    pos_y = transmitter_pos[:,1].view(S, 1, 1)

    Distance_2D = (x - pos_x) ** 2 + (y - pos_y) ** 2

    H_2 = (torch.tensor(DELTA_HEIGHT_LIST)**2).view(B, 1, 1, 1)

    Distance_3D = torch.clamp(torch.sqrt(Distance_2D.view(1, S, m ,n).repeat(B, 1, 1, 1) + H_2), min=1)
    p_1 = torch.tensor(p_1).view(1,S,1,1).repeat(B, 1, m, n)

    if LogPathLoss:
        radio_map_raw = torch.log10(p_1) - PLE * torch.log10(Distance_3D) + noise_std * torch.randn(Distance_3D.size())
        radio_map_db = torch.sum(radio_map_raw, dim=1).clamp(-1,1)
    else:
        radio_map_raw = p_1 / Distance_3D ** PLE 
        radio_map_db = torch.log10(torch.sum(radio_map_raw, dim=1)).clamp(-1,1)
        radio_map_db = (radio_map_db + noise_std * torch.randn(radio_map_db.size())).clamp(-1,1)

    radio_map_db[building >= 1] = -1

    return radio_map_db

def get_transmitter_pos(transmitter_pos_mat):
    flat_arr = transmitter_pos_mat.flatten()
    top_indices = np.argpartition(-flat_arr, 100)[:100]
    X = np.column_stack(np.unravel_index(top_indices, transmitter_pos_mat.shape))
    centers, _ = mean_shift(X, bandwidth=10)
    return centers


def get_transmitter_pos_fast(transmitter_pos_mat):
    flat_arr = transmitter_pos_mat.flatten()
    top_indices = np.argpartition(-flat_arr, 100)[:100]
    X = np.column_stack(np.unravel_index(top_indices, transmitter_pos_mat.shape))

    X = np.unique(X, axis=0)

    bandwidth = 10  # can be change

    centers, _ = mean_shift(
        X, 
        bandwidth=bandwidth,
        bin_seeding=True,
        min_bin_freq=1,
        max_iter=100,
        n_jobs=-1
    )
    return centers



from scipy.interpolate import Rbf

def rbf(point_cloud, building_0):
    x_grid, y_grid = np.mgrid[0:128, 0:128]

    rbf = Rbf(point_cloud[:,0], point_cloud[:,1], point_cloud[:,3], function='gaussian', smooth=1e-3)
    interpolated_values = rbf(x_grid, y_grid)

    interpolated_values_clipped = np.clip(interpolated_values, -1, 1)
    interpolated_values_clipped[building_0 >= 1] = -1

    return interpolated_values_clipped

def rbf_mix(point_cloud, building_0):
    x_grid, y_grid = np.mgrid[0:128, 0:128]

    rbf = Rbf(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], point_cloud[:,3], function='gaussian', smooth=1e-3)
    interpolated_values = rbf(x_grid, y_grid, np.ones_like(x_grid)*1.5).reshape(128,128)

    interpolated_values_clipped = np.clip(interpolated_values, -1, 1)
    interpolated_values_clipped[building_0 >= 1] = -1

    return interpolated_values_clipped


def get_phy_radiomap(point_cloud, building, point_cloud_height):

    phy_radiomap_list = []
    for point, b  in zip(point_cloud, building):
        radiomap_CZ = rbf(point.cpu(), b[point_cloud_height].cpu())
        pos = get_transmitter_pos(radiomap_CZ)
        n, p_1 = get_transmitter_info(pos, point.cpu())
        phy_radiomap = pos2radiomap(pos, p_1, b.cpu(), n)
        if torch.any(torch.isnan(phy_radiomap)):
            print("!!!There is NAN here, check the data!!!")
            print(p_1,pos)
        phy_radiomap_list.append(phy_radiomap)

    return torch.stack(phy_radiomap_list).to(torch.float32) #  ,pos


# to accelerate training inference speed, precomputed hata shift values are stored.
ENV2GEO={5: 'urban', 6: 'urban', 7: 'urban', 8: 'urban', 3: 'water', 4: 'water', 1: 'nature', 2: 'nature', 9: 'nature', 10: 'nature', 11: 'nature'}
SHIFT_LIST = {'urban':[[0, 0.02352941176470591, 0.015686274509803866], [-0.02352941176470591, 0, 0.007843137254901933], [-0.015686274509803866, -0.007843137254901933, 0]],
              'water':[[0, 0, -0.015686274509803866], [0, 0, -0.015686274509803866], [0.015686274509803866, 0.015686274509803866, 0]],
              'nature':[[0, 0.015686274509803977, 0.015686274509803866], [-0.015686274509803977, 0, 0], [-0.015686274509803866, 0, 0]]}
DROP_OUT = [-1, -0.388235294117647, -0.3019607843137255]

W_phy = []

def get_height_num(n):
    if n < 1:
        return 0
    elif n <10:
        return 1
    else:
        return 2
    
def get_phy_radiomap_mix_pc(point_cloud, building, env):
    phy_radiomap = torch.ones_like(building)*-1
    
    for n, points in enumerate(point_cloud):
        if len(env.shape) == 0:
            shift = SHIFT_LIST[ENV2GEO[int(env.item())]]
        else:
            shift = SHIFT_LIST[ENV2GEO[int(env[n])]]

        
        for p in points:
            if p[3] == -1 or (get_height_num(p[2]) > 0 and DROP_OUT[get_height_num(p[2])] > p[3]):
                continue
            phy_radiomap[n,:,int(p[0]),int(p[1])] = p[3] + torch.tensor(shift[get_height_num(p[2])])

    phy_radiomap[building == 1] = -1

    return phy_radiomap.to(torch.float32)


def get_phy_radiomap_hata(sample_map, building, env, target_height):
    N = sample_map.shape[0]

    phy_map = sample_map[:,target_height:target_height+1]

    if len(env.shape) == 0:
        shift = SHIFT_LIST[ENV2GEO[int(env.item())]]
    else:
        shift = SHIFT_LIST[ENV2GEO[int(env[0])]]

    for layer_num in range(3):
        if layer_num == target_height:
            pass
        
        shift_mat_mask = torch.logical_and(sample_map[:,layer_num:layer_num+1] > DROP_OUT[layer_num], phy_map > -1)
        shift_mat = sample_map[:,layer_num:layer_num+1] + shift_mat_mask*shift[layer_num][target_height]

        phy_map[shift_mat_mask] = shift_mat[shift_mat_mask]

    phy_map[building[:,target_height:target_height+1] == 1] = -1

    return phy_map.to(torch.float32)


def get_phy_radiomap_mix(pc, sample_map, building, env, target_height, H = 10, fast=False):
    if fast:
        # Ignore the F model to speed up, If FT need be True
        return get_phy_radiomap_hata(sample_map, building, env, target_height)
    
    hata = get_phy_radiomap_hata(sample_map, building, env, target_height)
    f = get_phy_radiomap(pc,building,target_height)[:,target_height:target_height+1,:,:]
    f[hata == -1] = -1

    w = pow(2,-target_height/H)

    return hata*w + f.to(hata.device)*(1-w)
