import torch
from tqdm import tqdm

from ddim.ddpm import DDPM

import numpy as np


class DDIM(DDPM):

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

    def sample_backward(self,
                        img_or_shape,
                        net, condition_without_noise,
                        device,
                        simple_var=False,
                        ddim_step=10,
                        eta=0,
                        save_video_path = "",
                        print_tqdm=True):

        ts = torch.linspace(self.n_steps, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)
            
        batch_size = x.shape[0]
        net = net.to(device)

        if save_video_path != "":
            video = [x]

        if print_tqdm:
            for i in tqdm(range(1, ddim_step + 1),
                        f'DDIM sampling with eta {eta} simple_var {simple_var}'):
                cur_t = ts[i - 1] - 1
                prev_t = ts[i] - 1

                ab_cur = self.alpha_bars[cur_t]
                ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1

                t_tensor = torch.tensor([cur_t] * batch_size,
                                        dtype=torch.long).to(device)
                
                condition_with_noise = torch.cat((x, condition_without_noise),dim=1)
                eps = net(condition_with_noise, t_tensor)
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x)

                first_term = (ab_prev / ab_cur)**0.5 * x
                second_term = ((1 - ab_prev - var)**0.5 -
                            (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
                x = first_term + second_term + third_term
                
                if save_video_path != "":
                    video.append(x)
        else:
            for i in range(1, ddim_step + 1):
                cur_t = ts[i - 1] - 1
                prev_t = ts[i] - 1

                ab_cur = self.alpha_bars[cur_t]
                ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1

                t_tensor = torch.tensor([cur_t] * batch_size,
                                        dtype=torch.long).to(device).unsqueeze(1)
                
                condition_with_noise = torch.cat((x, condition_without_noise),dim=1)
                eps = net(condition_with_noise, t_tensor)
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x)

                first_term = (ab_prev / ab_cur)**0.5 * x
                second_term = ((1 - ab_prev - var)**0.5 -
                            (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
                x = first_term + second_term + third_term
                
                if save_video_path != "":
                    video.append(x)

        if save_video_path != "":
            np.save(save_video_path,torch.stack(video).cpu())

        return x