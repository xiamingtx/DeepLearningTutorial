#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:MINSTDataSetExample.py
# author:xm
# datetime:2023/2/12 18:58
# software: PyCharm

"""
演示如何通过torchvision的datasets模块 进行数据集的加载
"""

# import module your need
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# prepare dataset
train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32,
                         shuffle=False)

# define model

# training cycle:  forward, backward, update
if __name__ == '__main__':
    for epoch in range(100):
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # to do
            pass
