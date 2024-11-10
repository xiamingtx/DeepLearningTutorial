#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:data_loader.py
# author:xm
# datetime:2024/7/12 14:38
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from torch.utils.data import DataLoader, Dataset
from dataset import Pinwheel, Checkerboard
from utils import show_2d_data


def create_data_loader(dataset: Dataset, batch_size: int, shuffle: bool = False,
                       num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )


if __name__ == '__main__':
    data_size = 2 ** 26
    pinwheel_dataset = Pinwheel(data_size)
    checkerboard_dataset = Checkerboard(size=data_size, grid_size=8)
    batch_size = 2 ** 16
    pinwheel_data_loader = create_data_loader(pinwheel_dataset, batch_size, num_workers=0)
    checkerboard_data_loader = create_data_loader(checkerboard_dataset, batch_size, num_workers=0)

    pinwheel_batch = next(iter(pinwheel_data_loader))
    checkerboard_batch = next(iter(checkerboard_data_loader))
    show_2d_data(checkerboard_batch)
