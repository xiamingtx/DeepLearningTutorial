#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:img_encoder.py
# author:xm
# datetime:2024/3/18 20:08
# software: PyCharm

"""
ResNet for Image Encoding
"""

# import module your need
import torch
from torch import nn
import torch.nn.functional as F


class ReisdualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ReisdualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        out = self.relu(x + self.shortcut(residual))
        return out


class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        self.res_block1 = ReisdualBlock(in_channels=1, out_channels=16, stride=2)  # (bs, 16, 14, 14)
        self.res_block2 = ReisdualBlock(in_channels=16, out_channels=4, stride=2)  # (bs, 4, 7, 7)
        self.res_block3 = ReisdualBlock(in_channels=4, out_channels=1, stride=2)  # (bs, 1, 4, 4)
        self.fc = nn.Linear(in_features=16, out_features=8)
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.fc(torch.flatten(x, 1))
        out = self.ln(x)
        return out


if __name__ == '__main__':
    img_encoder = ImgEncoder()
    input = torch.randn(1, 1, 28, 28)
    out = img_encoder(input)
    print(out.shape)
