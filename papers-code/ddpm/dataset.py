#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:dataset.py
# author:xm
# datetime:2024/3/20 21:20
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def get_fashion_mnist_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
