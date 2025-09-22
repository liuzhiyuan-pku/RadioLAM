import os
import time

import cv2
import einops
import torch
import torch.nn as nn
import random

from configs import FT_CONFIG, UNet_cfg, EXP_UNet_cfg
from data_process.dataset import get_dataloader
from data_process.tools import get_context_device
from ddim.ddpm import DDPM
from model.UNet import MoG_FT
from model.UDiT import U_DiT

from accelerate import Accelerator

from model.phy_model import get_phy_radiomap_mix

from tqdm import *

from lion_pytorch import Lion


def train(ddpm: DDPM,
          net,
          dataset_type,
          scheduler_cfg,
          target_height,
          batch_size=512,
          n_epochs=50,
          device='cuda',
          ckpt_path='model.pth',
          sparse_num=100):
    print('batch size:', batch_size)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(dataset_type,
                                batch_size,
                                'mix',
                                sparse_num,
                                channel_num=FT_CONFIG['channel_num'],
                                Geographic_type=FT_CONFIG['Geographic_type'])

    loss_fn = nn.MSELoss()

    # optimizer = torch.optim.Adam(net.parameters(), scheduler_cfg['lr'], weight_decay=0.01)
    optimizer = Lion(net.parameters(),
                     lr=scheduler_cfg['lr'],  # 初始学习率（通常比 Adam 小 3-10 倍）
                     weight_decay=1e-2,  # 权重衰减（类似 AdamW）
                     betas=(0.9, 0.99)  # 动量参数（论文推荐 (0.9, 0.99)）
                     )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg['milestones'],
                                                gamma=scheduler_cfg['gamma'])

    net, optimizer, dataloader, scheduler = accelerator.prepare(
        net, optimizer, dataloader, scheduler
    )

    for n in range(len(net.GEN_list)):
        net.GEN_list[n] = net.GEN_list[n].to(device)

    tic = time.time()

    best_loss = 100
    for e in range(n_epochs):
        net.train()
        total_loss = 0

        scheduler.step()
        for param_group in optimizer.param_groups:
            print("learning rate", param_group['lr'])

        for pc, sample_map, building, terrain, frequencies, env, radiomap in tqdm(dataloader,
                                                                              disable=not accelerator.is_local_main_process):
            current_batch_size = radiomap.shape[0]

            building, terrain, radiomap = \
                building.to(device), terrain.to(device), radiomap.to(device)


            if target_height == 'X':
                height_num = random.randint(0, 2)
            else:
                height_num = target_height

            # Change the value of fast to False when approaching convergence
            phy_radiomap = get_phy_radiomap_mix(pc, sample_map.cpu(), building.cpu(), env, height_num, fast = True).to(device)

            x = radiomap[:, height_num:height_num + 1]

            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)

            context = get_context_device(height_num, current_batch_size, 256, device=device).repeat(1, 32).view(current_batch_size, 1,
                                                                                                 128, 128)

            condition_with_noise = torch.cat(
                (x_t, phy_radiomap, context, building[:, height_num:height_num + 1] * 2 - 1, terrain * 2 - 1), dim=1)
            eps_theta = net(condition_with_noise, t)

            loss = loss_fn(eps_theta, eps)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item() * current_batch_size

        local_total_loss = torch.tensor(total_loss, device=accelerator.device)
        total_loss = accelerator.reduce(local_total_loss, reduction="sum").item()

        if accelerator.is_local_main_process:
            total_loss /= len(dataloader.dataset)
            toc = time.time()
            print(f'epoch {e + 1} loss: {total_loss} elapsed {(toc - tic):.2f}s')

            if total_loss < best_loss:
                best_loss = total_loss
                best_epoch = e + 1
                print(f"Saving best model at Epoch {best_epoch} with Loss: {best_loss:.5f}")
                net.save_model(accelerator,ckpt_path, FT_CONFIG['channel_num'], best_epoch, target_height, sparse_num, best_loss)

    accelerator.end_training()
    print('Done')


if __name__ == '__main__':
    if FT_CONFIG['point_cloud_height'] != 'mix':
        assert 0, "FT_CONFIG is not MIX!!!"

    os.makedirs('work_dirs', exist_ok=True)

    n_steps = 1000

    accelerator = Accelerator()
    device = accelerator.device

    model_path = FT_CONFIG['model_path']

    net = MoG_FT(len(FT_CONFIG['weight_path']), n_steps, UNet_cfg, EXP_UNet_cfg, FT_CONFIG['router_path'], FT_CONFIG['weight_path'])


    ddpm = DDPM(device, n_steps)

    train(ddpm,
          net,
          FT_CONFIG['dataset_type'],
          FT_CONFIG['scheduler_cfg'],
          FT_CONFIG['target_height'],
          batch_size=FT_CONFIG['batch_size'],
          n_epochs=FT_CONFIG['n_epochs'],
          device=device,
          ckpt_path=model_path,
          sparse_num=FT_CONFIG['sparse_num'])
