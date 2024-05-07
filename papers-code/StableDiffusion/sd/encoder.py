#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:encoder.py
# author:xm
# datetime:2024/5/5 23:59
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAEAttentionBlock, VAEResidualBlock


class VAEEncoder(nn.Sequential):
    def __init__(self):
        super(VAEEncoder, self).__init__(
            # (bs, c, h, w) -> (bs, 128, h, w)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (bs, 128, h, w) -> (bs, 128, h, w)
            VAEResidualBlock(128, 128),
            # (bs, 128, h, w) -> (bs, 128, h, w)
            VAEResidualBlock(128, 128),

            # (bs, 128, h, w) -> (bs, 128, h / 2, w / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (bs, 128, h / 2, w / 2) -> (bs, 256, h / 2, w / 2)
            VAEResidualBlock(128, 256),
            # (bs, 256, h / 2, w / 2) -> (bs, 256, h / 2, w / 2)
            VAEResidualBlock(256, 256),

            # (bs, 256, h / 2, w / 2) -> (bs, 256, h / 4, w / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (bs, 256, h / 4, w / 4) -> (bs, 512, h / 4, w / 4)
            VAEResidualBlock(256, 512),
            # (bs, 512, h / 4, w / 4) -> (bs, 512, h / 4, w / 4)
            VAEResidualBlock(512, 512),

            # (bs, 512, h / 4, w / 4) -> (bs, 512, h / 8, w / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (bs, 512, h / 8, w / 8) -> (bs, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (bs, 512, h / 8, w / 8) -> (bs, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (bs, 512, h / 8, w / 8) -> (bs, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),

            # (bs, 512, h / 8, w / 8) -> (bs, 512, h / 8, w / 8)
            VAEAttentionBlock(512),
            # (bs, 512, h / 8, w / 8) -> (bs, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (bs, 512, h / 8, w / 8) -> (bs, 512, h / 8, w / 8)
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.SiLU(),

            # (bs, 512, h / 8, w / 8) -> (bs, 8, h / 8, w / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (bs, 8, h / 8, w / 8) -> (bs, 8, h / 8, w / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (bs, c, h, w), noise: (bs, out_channels, h / 8, w / 8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (bs, 8, h / 8, w / 8) -> two tensors of shape (bs, 4, h / 8, w / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # (bs, 4, h / 8, w / 8) -> (bs, 4, h / 8, w / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (bs, 4, h / 8, w / 8) -> (bs, 4, h / 8, w / 8)
        variance = log_variance.exp()
        # (bs, 4, h / 8, w / 8) -> (bs, 4, h / 8, w / 8)
        stdev = variance.sqrt()

        # Z=N(0, 1) -> N(mean, variance)=X?
        # X = mean + stdev * Z
        x = mean + stdev * noise
        # Scale the output by a constant
        x *= 0.18215
        return x


if __name__ == '__main__':
    encoder = VAEEncoder()
    x = torch.randn((1, 3, 500, 500))
    noise = torch.randn((1, 4, 500 // 8, 500 // 8))
    print(encoder(x, noise))
