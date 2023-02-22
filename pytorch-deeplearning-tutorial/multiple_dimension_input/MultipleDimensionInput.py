#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:MultipleDimensionInput.py
# author:xm
# datetime:2023/2/12 11:47
# software: PyCharm

"""
处理多维特征的输入
理解矩阵是对空间的映射变换
"""

# import module your need
import torch
import numpy as np
from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

# prepare dataset
# 如果是np.loadtxt 如果传入的是.gz压缩包 也可以解析
# xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
xy = np.loadtxt('../data/diabetes.csv', delimiter=',', dtype=np.float32)

x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


# define model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入8维 输出6维
        self.linear2 = torch.nn.Linear(6, 4)  # 输入6维 输出4维
        self.linear3 = torch.nn.Linear(4, 1)  # 输入4维 输出1维
        # self.activate = torch.nn.ReLU()  # 我们可以尝试使用不同的激活函数
        self.activate = torch.nn.Sigmoid()  # 这里Sigmoid是作为一层 一个运算模块

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []

# training cycle forward, backward, update
for epoch in range(100):
    # forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()

    # update
    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
