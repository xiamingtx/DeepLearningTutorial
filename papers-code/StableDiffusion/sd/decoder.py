#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:decoder.py
# author:xm
# datetime:2024/5/6 0:00
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super(VAEAttentionBlock, self).__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, c, h, w)
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape

        # (bs, c, h, w) -> (bs, c, h * w)
        x = x.view(n, c, h * w)
        # (bs, c, h * w) -> (bs, h * w, c)
        x = x.transpose(-1, -2)

        # (bs, h * w, c) -> (bs, h * w, c)
        x = self.attention(x)
        # (bs, h * w, c) -> (bs, c, h * w)
        x = x.transpose(-1, -2)
        # (bs, c, h * w) -> (bs, c, h, w)
        x = x.view(n, c, h, w)

        x += residue

        return x


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VAEResidualBlock, self).__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, in_channels, h, w)
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAEDecoder(nn.Sequential):
    def __init__(self):
        super(VAEDecoder, self).__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            # (bs, 512, h / 8, w / 8) -> (bs, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),

            # (bs, 512, h / 8, w / 8) -> (bs, 512, h / 4, w / 4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),

            # (bs, 512, h / 4, w / 4) -> (bs, 512, h / 2, w / 2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),

            # (bs, 512, h / 2, w / 2) -> (bs, 512, h, w)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # (bs, 512, h, w) -> (bs, 3, h, w)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, 4, h / 8, w / 8)
        x /= 0.18215

        for module in self:
            x = module(x)

        # (bs, 3, h, w)
        return x
