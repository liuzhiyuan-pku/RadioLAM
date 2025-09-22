import torch
import torch.nn as nn
import math


env_classes_num = 11
Height_list = torch.tensor([15,200,2000])

def value_embedding(heightsteps, dim, max_period=3000):
    """
    Create sinusoidal timestep embeddings.
    :param heightsteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=heightsteps.device)
    args = heightsteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


def get_context_device(height_num, batch_size, dim=64, device = 'cuda'):
    # 获取高度特征向量
    height = Height_list[height_num]
    height_vector = value_embedding(height.view(-1), dim*2, 5000).to(device).repeat(batch_size,1)

    # feq_vector = value_embedding(frequencies.view(-1), dim).to(device)

    # env_emb = nn.Embedding.from_pretrained(torch.load('/root/autodl-tmp/RadioMoG/env.pth'), freeze=True).to(device)
    # env_vector = env_emb(env.to(torch.int))

    # return torch.cat([height_vector, env_vector], dim = 1)
    return height_vector