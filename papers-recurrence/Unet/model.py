#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:model.py
# author:xm
# datetime:2024/3/9 17:16
# software: PyCharm

"""
follow this video:
https://www.bilibili.com/video/BV17z421R7BR
"""

# import module your need
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Unet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channel, feature))
            in_channel = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(in_channel=feature * 2, out_channel=feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if skip_connection.shape != x.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat = torch.cat((x, skip_connection), dim=1)
            x = self.ups[idx + 1](concat)
        return self.final_conv(x)


def test():
    x = torch.randn((3, 3, 160, 160))
    model = Unet(in_channel=3, out_channel=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)


if __name__ == '__main__':
    test()
