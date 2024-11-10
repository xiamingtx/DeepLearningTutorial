#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:utils.py
# author:xm
# datetime:2024/7/12 14:34
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from torch import Tensor
import matplotlib.pyplot as plt


def align_shape(x: Tensor, coeff):
    """ 将 coeff 的维度与 x 对齐. """

    if isinstance(coeff, dict):
        for k, v in coeff.items():
            if isinstance(v, Tensor):
                while len(v.shape) < len(x.shape):
                    v = v.unsqueeze(-1)
                coeff[k] = v
    elif isinstance(coeff, Tensor):
        while len(coeff.shape) < len(x.shape):
            coeff = coeff.unsqueeze(-1)

    return coeff


def show_2d_data(data: Tensor):
    plt.figure(figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    plt.show()
    plt.close()
