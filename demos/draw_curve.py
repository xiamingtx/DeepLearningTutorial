#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:draw_curve.py
# author:xm
# datetime:2024/11/6 9:24
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm

# 定义参数
max_threshold = 0.5
warmup_epochs = 50
max_epochs = 200

# 定义一个用于计算阈值的函数
def compute_threshold(epoch, warmup_epochs, max_epochs, max_threshold):
    initial_threshold = 1 / 10
    if epoch < warmup_epochs:
        warmup_ratio = epoch / warmup_epochs
        threshold = initial_threshold + warmup_ratio * (max_threshold - initial_threshold)
    else:
        current_progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        cos_out = (math.cos(current_progress * math.pi) + 1) / 2
        threshold = max_threshold * (1 - cos_out)
    return threshold

# 生成所有的 epoch 值
epochs = np.arange(0, max_epochs + 1)

# 计算每个 epoch 对应的 threshold
thresholds = [compute_threshold(epoch, warmup_epochs, max_epochs, max_threshold) for epoch in epochs]

# 绘制曲线
plt.figure(figsize=(10, 6))  # 设置图表大小


colors = cm.plasma(np.linspace(0, 1, len(epochs)))
plt.plot(epochs, thresholds, color=colors[-1], linewidth=2.5, alpha=0.8, label='Threshold Curve')

plt.xlabel('Epochs')
plt.ylabel('Threshold')
# 展示图表
plt.tight_layout()  # 自动调整布局以适应标签
plt.show()
