import os
import time

import cv2
import einops
import torch
import torch.nn as nn

from configs import MoG_TEST_CONFIG, UNet_cfg, EXP_UNet_cfg
from data_process.dataset import get_dataloader
from data_process.tools import get_context_device

from ddim.ddim import DDIM
from ddim.ddpm import DDPM

from model.MMDiT import MMDiTX
from model.UDiT import U_DiT
from model.phy_model import get_phy_radiomap_mix
from model.UNet import MoG_infer

from accelerate import Accelerator

import torch.nn.functional as F

from tqdm import *

import matplotlib.pyplot as plt
import re


Geographic_items = {'urban': ['05.Suburban', '06.DenseUrban', '07.Rural', '08.OrdinaryUrban'],
                        'water': ['03.Ocean', '04.Lake'],
                        'nature': ['01.Grassland','02.Island','09.Desert','10.Mountainous','11.Forest']}


def get_image_evaluation(result_list: list, target_list: list, time_cost=0, if_return_str=True):
    '''
    pip install pytorch_msssim -i https://pypi.tuna.tsinghua.edu.cn/simple
    '''
    from pytorch_msssim import ssim

    mse_loss_fn = nn.MSELoss()

    result_list = torch.concat(result_list, dim=0).view(-1, 1, 128, 128)
    target_list = torch.concat(target_list, dim=0).view(-1, 1, 128, 128)

    evaluation_val = {}

    mae_loss_fn = nn.L1Loss()
    evaluation_val['MAE'] = mae_loss_fn(result_list, target_list)
    evaluation_val['MSE'] = mse_loss_fn(result_list, target_list)
    evaluation_val['RMSE'] = torch.sqrt(evaluation_val['MSE'])
    evaluation_val['NMSE'] = evaluation_val['MSE'] / mse_loss_fn(target_list * 0, target_list)
    evaluation_val['PSNR'] = 20 * torch.log10(1 / evaluation_val['RMSE'])

    evaluation_val['SSIM'] = ssim(result_list, target_list, data_range=1, size_average=True)

    if if_return_str:
        s = f'Test Dataset Size = {result_list.shape[0]}\ntime elapsed = {time_cost:.2f}s'
        for k, v in evaluation_val.items():
            s += f'\n{k:>6} value = {v:.6}'
        return s
    else:
        s = f'Test Dataset Size = {result_list.shape[0]}\ntime elapsed = {time_cost:.2f}s'
        for k, v in evaluation_val.items():
            s += f'\n{k:>6} value = {v:.6}'
        return s, evaluation_val


def test_imgs(ddim:DDIM,
              accelerator,
              net,
              dataset_type,
              Geographic_type,
              target_height,
              sample_num=16,
              step=10,
              device='cuda',
              sparse_num=100):
    print('sample_num:', sample_num)
    print("sparse_num:", sparse_num)

    dataloader = get_dataloader(dataset_type, 1,
                                'mix',
                                sparse_num,
                                channel_num=MoG_TEST_CONFIG['channel_num'],
                                Geographic_type=Geographic_type)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    sp_loss_fn = nn.MSELoss(reduction='none')

    sigma = 0
    MAX_R = 0.05
    sigma_m = 0.1

    net.eval()

    result_list = []
    target_list = []

    net, dataloader = accelerator.prepare(net, dataloader)

    with torch.no_grad():
        epoch_samples = 0

        tic = time.time()

        for pc, sample_map, building, terrain, frequencies, env, radiomap in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            epoch_samples += 1

            building, terrain = building.repeat(sample_num, 1, 1, 1), terrain.repeat(sample_num, 1, 1, 1)

            height_num = target_height

            # First, verify the results with fast=True, and continue fine-tuning only when the performance is good enough
            phy_radiomap = get_phy_radiomap_mix(pc, sample_map, building[0:1].cpu(),env, target_height, fast=True).to(device)
            phy_radiomap = phy_radiomap.repeat(sample_num, 1, 1, 1)

            context = get_context_device(height_num, 1, 256, device=device).repeat(sample_num,32).view(sample_num,1,128,128)

            x = (radiomap[:, height_num:height_num + 1] + 1) / 2

            cond = torch.cat((phy_radiomap, context, building[:, height_num:height_num+1] * 2 - 1, terrain * 2 - 1), dim=1)
            
            net.sigma = sigma
            imgs_list = ddim.sample_backward((sample_num, 1, 128, 128),
                            net, cond,
                            ddim_step = step,
                            device=device,
                            simple_var=False,
                            print_tqdm=False)

            imgs_list = imgs_list.clamp(-1, 1)

           
            mask_sparse_images = phy_radiomap[:, height_num:height_num + 1] == -1
            sp_loss = torch.sum(sp_loss_fn(torch.masked_fill(imgs_list, mask_sparse_images, -1),
                                            phy_radiomap[:, height_num:height_num + 1]).view(sample_num, -1), 1)
            
            R_div = torch.var(sp_loss)
            if R_div > MAX_R:
                sigma /= 2
            elif sigma < sigma_m:
                sigma += 0.01
            
            sigma = min(sigma,sigma_m)

            imgs_MSE = (torch.index_select(imgs_list, dim=0, index=torch.argmin(sp_loss)) + 1) / 2

            imgs_MSE = accelerator.gather(imgs_MSE)
            x = accelerator.gather(x)
            
            result_list.append(imgs_MSE)
            target_list.append(x)

                    
    toc = time.time()
    if accelerator.is_local_main_process:
        print("---------------------------------------------")
        s = get_image_evaluation(result_list, target_list, time_cost=(toc - tic), if_return_str=True)
        print(s)
        f = open(f'work_dirs/{Geographic_type}_{MoG_TEST_CONFIG["channel_num"]}H{target_height}S{sparse_num}.txt', 'w')
        f.write(s)
        f.close()

        print('Done')




if __name__ == '__main__':

    os.makedirs('work_dirs', exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device
    n_steps = 1000

    weight_path = list(map(lambda x: x[1], MoG_TEST_CONFIG['weight_path']))
    gen_num = len(weight_path)

    net = MoG_infer( gen_num, n_steps, UNet_cfg, EXP_UNet_cfg)
    net.load(MoG_TEST_CONFIG['router_path'], weight_path[0], weight_path[1:])

    ddim = DDIM(device, n_steps)

    for sp in [50]:
        for Geographic_type in Geographic_items['urban']:
            for target_height in [0, 1, 2]:
                test_imgs(ddim,
                        accelerator,
                        net,
                        MoG_TEST_CONFIG['dataset_type'],
                        Geographic_type,
                        target_height,
                        device=device,
                        step=10,
                        sample_num=MoG_TEST_CONFIG['sample_num'],
                        sparse_num=sp)
