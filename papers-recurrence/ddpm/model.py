#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:model.py
# author:xm
# datetime:2024/3/20 19:47
# software: PyCharm

"""
http://www.egbenz.com/#/my_article/19
"""

# import module your need
from functools import partial

import torch
from einops import rearrange
from torch import nn
import math
from utils import exists, default, Residual, Downsample, Upsample
from components import SinusoidalPositionEmbeddings, ConvNextBlock, ResnetBlock, LinearAttention, PreNorm, Attention
from forward import q_sample
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, kernel_size=7, padding=3)

        # [init_dim, dim, dim * 2, dim * 4, dim * 8] (default)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # [(init_dim, dim), (dim, dim * 2), (dim * 2, dim * 4), (dim * 4, dim * 8)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            # apply positional encoding and mlp
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        """
        :param x: noisy image, shape = (bs, c, h, w)
        :param time: timestep, shape = (bs,)
        :return: tensor with shape = (bs, c, h, w)
        """
        # apply conv, map input to init_dim channels, (bs, c, h, w) -> (bs, init_dim, h, w)
        x = self.init_conv(x)

        # employ positional encoding for timestep, (bs, ) -> (bs, dim) -> (bs, time_dim)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample stage
        # two ResNet/ConvNeXT modules + grouping normalization + attention module + residual linking + downsampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck, two blocks with attention module in middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample stage
        # two ResNet/ConvNeXT modules + grouping normalization + attention module + residual linking + upsampling
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


def p_losses(denoise_model, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    # if noise is non-given, sample noise from a standard gaussian distribution
    if noise is None:
        noise = torch.randn_like(x_start)

    # add noise to image according to the timestep=t
    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
