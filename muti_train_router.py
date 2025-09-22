import os
import time

import cv2
import einops
import torch
import torch.nn as nn
import random

from configs import ROUTER_CONFIG, UNet_cfg, EXP_UNet_cfg
from data_process.dataset import get_dataloader
from data_process.tools import get_context_device
from ddim.ddpm import DDPM
from model.UNet import Router
from model.UDiT import U_DiT

from accelerate import Accelerator

from model.phy_model import get_phy_radiomap_mix

from tqdm import *

from lion_pytorch import Lion


def train(net,
          dataset_type,
          scheduler_cfg,
          batch_size=512,
          n_epochs=50,
          device='cuda',
          ckpt_path='model.pth',):
    print('batch size:', batch_size)
    dataloader = get_dataloader(dataset_type,
                                batch_size,
                                'mix',
                                50,
                                channel_num=ROUTER_CONFIG['channel_num'],
                                Geographic_type=ROUTER_CONFIG['Geographic_type'])

    loss_fn = nn.NLLLoss()

    optimizer = Lion(net.parameters(),
                     lr=scheduler_cfg['lr'], 
                     weight_decay=1e-2,
                     betas=(0.9, 0.99)
                     )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg['milestones'],
                                                gamma=scheduler_cfg['gamma'])

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

        # x为dense_images
        for pc, sample_map, building, terrain, frequencies, env, radiomap in tqdm(dataloader,
                                                                              disable=not accelerator.is_local_main_process):
            current_batch_size = radiomap.shape[0]

            building, terrain, radiomap = \
                building.to(device), terrain.to(device), radiomap.to(device)

            for height_num in range(3):

                if random.random() > 0.5:
                    target = env - 5
                else:
                    target = torch.ones_like(env) * (ROUTER_CONFIG['gen_num'] - 1)
                
                target = target.to(torch.long)

                context = get_context_device(height_num, current_batch_size, 256, device=device).repeat(1, 32).view(current_batch_size, 1,
                                                                                                     128, 128)

                condition = torch.cat(
                    (context, building[:, height_num:height_num + 1] * 2 - 1, terrain * 2 - 1), dim=1)
                out = net(condition)

                loss = loss_fn(torch.log(out), target)

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                total_loss += loss.item() * current_batch_size / 3

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
                                                                      f"ROUTER_{ROUTER_CONFIG['Geographic_type']}_{ROUTER_CONFIG['channel_num']}E{best_epoch}L{best_loss:.5f}.pth"))

            unwrapped_model = accelerator.unwrap_model(net)
            torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_path, "model_last.pth"))

    accelerator.end_training()
    print('Done')


if __name__ == '__main__':

    accelerator = Accelerator()
    device = accelerator.device

    model_path = ROUTER_CONFIG['model_path']

    net = Router(ROUTER_CONFIG['gen_num'])

    if ROUTER_CONFIG['pretrain_weight_path'] != "":
        net.load_state_dict(torch.load(ROUTER_CONFIG['pretrain_weight_path']))

    train(net,
          ROUTER_CONFIG['dataset_type'],
          ROUTER_CONFIG['scheduler_cfg'],
          batch_size=ROUTER_CONFIG['batch_size'],
          n_epochs=ROUTER_CONFIG['n_epochs'],
          device=device,
          ckpt_path=model_path)
