#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:diffusion.py
# author:xm
# datetime:2024/5/6 12:40
# software: PyCharm

"""
tutorial: https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=4626s
code: https://github.com/hkproj/pytorch-stable-diffusion
"""

# import module your need
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super(TimeEmbedding, self).__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        # (1, 1280)
        return x


class UNETResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super(UNETResidualBlock, self).__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature: (bs, in_channels, h, w)
        # time: (1, 1280)
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNETAttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context: int = 768):
        super(UNETAttentionBlock, self).__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: (bs, features, h, w)
        # context: (bs, seq_len, dim)
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        # (bs, features, h, w) -> (bs, features, h * w)
        x = x.view(n, c, h * w)
        # (bs, features, h * w) -> (bs, h * w, features)
        x = x.transpose(-1, -2)

        # Normalization + Self Attention with skip connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # Normalization + Cross Attention with skip connection
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        # Normalization + FeedForward with GeGLU and skip Connection
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        # (bs, h * w, features) -> (bs, features, h * w)
        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (bs, features, h, w) -> (bs, features, h * 2, w * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNETAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNETResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.encoders = nn.ModuleList([
            # (bs, 4, h / 8, w / 8) -> (bs, 320, h / 8, w / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),
            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),

            # (bs, 320, h / 8, w / 8) -> (bs, 320, h / 16, w / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNETResidualBlock(320, 640), UNETAttentionBlock(8, 80)),
            SwitchSequential(UNETResidualBlock(640, 640), UNETAttentionBlock(8, 80)),

            # (bs, 640, h / 16, w / 16) -> (bs, 640, h / 32, w / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNETResidualBlock(640, 1280), UNETAttentionBlock(8, 160)),
            SwitchSequential(UNETResidualBlock(1280, 1280), UNETAttentionBlock(8, 160)),

            # (bs, 1280, h / 32, w / 32) -> (bs, 1280, h / 64, w / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNETResidualBlock(1280, 1280)),
            SwitchSequential(UNETResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNETResidualBlock(1280, 1280),
            UNETAttentionBlock(8, 160),
            UNETResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # (bs, 2560, h / 64, w / 64) -> (bs, 1280, h / 64, w / 64)
            SwitchSequential(UNETResidualBlock(2560, 1280)),
            SwitchSequential(UNETResidualBlock(2560, 1280)),

            SwitchSequential(UNETResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)),
            SwitchSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)),

            SwitchSequential(UNETResidualBlock(1920, 1280), UNETAttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UNETResidualBlock(1920, 640), UNETAttentionBlock(8, 80)),
            SwitchSequential(UNETResidualBlock(1280, 640), UNETAttentionBlock(8, 80)),

            SwitchSequential(UNETResidualBlock(960, 640), UNETAttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNETResidualBlock(960, 320), UNETAttentionBlock(8, 40)),
            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)),
            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)),

        ])

    def forward(self, x, context, time):
        # x: (bs, 4, Height / 8, Width / 8)
        # context: (bs, seq_len, dim)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNETOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNETOutputLayer, self).__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, 320, h / 8, w / 8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # (bs, 4, h / 8, w / 8)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNETOutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (bs, 4, h / 8, w / 8)
        # context: (bs, seq_len, dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (bs, 4, h / 8, w / 8) -> (bs, 320, h / 8, w / 8)
        output = self.unet(latent, context, time)

        # (bs, 320, h / 8, w / 8) -> (bs, 4, h / 8, w / 8)
        output = self.final(output)

        return output
