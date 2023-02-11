#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:OptimizersTest.py
# author:xm
# datetime:2023/2/11 18:27
# software: PyCharm

"""
尝试不同优化器
torch.optim.Adagrad
torch.optim.Adam
torch.optim.Adamax
torch.optim.ASGD
torch.optim.LBFGS
torch.optim.RMSprop
torch.optim.Rprop
torch.optim.SGD

"""

# import module your need
import torch
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning


x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):  # 构造函数
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 构造对象，并说明输入输出的维数，第三个参数默认为true，表示用到b

    def forward(self, x):
        y_pred = self.linear(x)  # 可调用对象，计算y=wx+b
        return y_pred


model = LinearModel()  # 实例化模型

criterion = torch.nn.MSELoss(reduction='sum')
# model.parameters()会扫描module中的所有成员，如果成员中有相应权重，那么都会将结果加到要训练的参数集合上
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)  # lr为学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # lr为学习率

epoch_list = []
mse_list = []

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    mse_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)


# 绘图
plt.plot(epoch_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
