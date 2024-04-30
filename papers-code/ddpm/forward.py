#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:forward.py
# author:xm
# datetime:2024/3/20 20:33
# software: PyCharm

"""
original paper: We set the forward process variances to constants increasing linearly from beta_1=1e-4 to beta_T=0.02
However, https://arxiv.org/abs/2102.09672 show that cosine is better
"""

# import module your need
import torch
import torch.nn.functional as F
from utils import extract


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start



# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    # add noise
    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image


if __name__ == '__main__':
    timesteps = 200

    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    from PIL import Image
    import requests

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    # image.show()

    from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

    image_size = 128
    # transform image to tensor and apply normalization (from {0, 1, ..., 255} to [-1, 1])
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Lambda(lambda t: (t * 2) - 1),
    ])

    x_start = transform(image).unsqueeze(0)
    # print(x_start.shape)  # torch.Size([1, 3, 128, 128])

    import numpy as np

    # reverse process
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    # convert tensor to image
    # reverse_transform(x_start.squeeze()).show()

    # take time step
    t = torch.tensor([40])

    # add noise to image and show it
    # get_noisy_image(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).show()

    # use seed for reproducability
    # torch.manual_seed(0)
    # visualize the sequence process
    # plot([get_noisy_image(image, x_start, torch.tensor([t]), sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    #       for t in [0, 50, 100, 150, 199]])
