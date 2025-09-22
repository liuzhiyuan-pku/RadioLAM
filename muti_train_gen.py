import os
import time

import cv2
import einops
import torch
import torch.nn as nn
import random

from configs import CONFIG, UNet_cfg, EXP_UNet_cfg
from data_process.dataset import get_dataloader
from data_process.tools import get_context_device
from ddim.ddpm import DDPM
from model.UNet import UNet
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
          sparse_num = 100):
    print('batch size:', batch_size)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(dataset_type,
                                batch_size,
                                'mix',
                                sparse_num,
                                channel_num=CONFIG['channel_num'],
                                Geographic_type=CONFIG['Geographic_type'])
    
    loss_fn = nn.MSELoss()

    optimizer = Lion(net.parameters(),
                    lr=scheduler_cfg['lr'],          
                    weight_decay=1e-2,  
                    betas=(0.9, 0.99) 
                )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg['milestones'], gamma=scheduler_cfg['gamma'])

    net, optimizer, dataloader, scheduler = accelerator.prepare(
        net, optimizer, dataloader, scheduler
    )

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
                height_num = random.randint(0,2)
            else:
                height_num = target_height

            phy_radiomap = get_phy_radiomap_mix(pc, sample_map.cpu(), building.cpu(), env, height_num, fast = True).to(device)

            x = radiomap[:, height_num:height_num+1]

            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)

            context = get_context_device(height_num, current_batch_size, 256,  device=device).repeat(1,32).view(current_batch_size,1,128,128)

            condition_with_noise = torch.cat((x_t, phy_radiomap, context, building[:, height_num:height_num+1]*2-1, terrain*2-1),dim=1)
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
                unwrapped_model = accelerator.unwrap_model(net)
                torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_path,
                                                                      f"DDIM_{CONFIG['Geographic_type']}_{CONFIG['channel_num']}E{best_epoch}H{target_height}S{sparse_num}L{best_loss:.5f}.pth"))

            unwrapped_model = accelerator.unwrap_model(net)
            torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_path, "model_last.pth"))

    accelerator.end_training()
    print('Done')



if __name__ == '__main__':
    if CONFIG['point_cloud_height'] != 'mix':
        assert 0, "CONFIG is not MIX!!!"

    os.makedirs('work_dirs', exist_ok=True)

    n_steps = 1000

    accelerator = Accelerator()
    device = accelerator.device

    model_path = CONFIG['model_path']

    if CONFIG['Geographic_type'] == 'urban':
        net = UNet(n_steps, UNet_cfg['img_shape'], UNet_cfg['channels'], UNet_cfg['pe_dim'],
                   UNet_cfg.get('with_attn', False), UNet_cfg.get('norm_type', 'ln'))
    else:
        net = UNet(n_steps, EXP_UNet_cfg['img_shape'], EXP_UNet_cfg['channels'], EXP_UNet_cfg['pe_dim'],
                       EXP_UNet_cfg.get('with_attn', False), EXP_UNet_cfg.get('norm_type', 'ln'))
    ddpm = DDPM(device, n_steps)

    if CONFIG['pretrain_weight_path'] != "":
        net.load_state_dict(torch.load(CONFIG['pretrain_weight_path']))


    train(ddpm,
          net,
          CONFIG['dataset_type'],
          CONFIG['scheduler_cfg'],
          CONFIG['target_height'],
          batch_size=CONFIG['batch_size'],
          n_epochs=CONFIG['n_epochs'],
          device=device,
          ckpt_path=model_path,
          sparse_num = CONFIG['sparse_num'])
