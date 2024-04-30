#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:dataset.py
# author:xm
# datetime:2024/4/30 17:27
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, Compose
import torchvision


class MNIST(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.ds = torchvision.datasets.MNIST('./mnist/', train=is_train, download=True)
        self.img_convert = Compose([
            PILToTensor(),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return self.img_convert(img) / 255.0, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = MNIST()

    img, label = ds[0]
    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
