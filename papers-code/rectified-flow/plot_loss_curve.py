#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:plot_loss_curve.py
# author:xm
# datetime:2024/10/2 15:03
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':

    # 画Loss曲线看收敛情况
    # 读取pth文件，获得loss_list
    checkpoint = torch.load('./checkpoints/v1.1-cfg/miniunet_49.pth')
    loss_list = checkpoint['loss_list']

    # 画图
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.tight_layout()
    plt.show()
