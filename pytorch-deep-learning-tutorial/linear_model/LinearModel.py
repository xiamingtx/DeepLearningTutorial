#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:linear_model.py
# author:xm
# datetime:2023/2/10 10:27
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import numpy as np
import matplotlib.pyplot as plt  # import necessary library to draw the graph

import warnings
warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

# prepare the train set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w  # Linear Model


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2  # Loss Function: MSE(Mean Square Error) 均方误差


# 穷举法
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):  # 生成权重 [0.0, 4.0] 每次间隔0.1
    print("w=", w)
    l_sum = 0
    # zip函数 将x和y根据索引连成元组数组返回 拿到的x_val和y_val就相当于横纵坐标了
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
