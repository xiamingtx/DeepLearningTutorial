#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:dataset.py
# author:xm
# datetime:2024/3/9 18:47
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('RGB'))

        return image, mask