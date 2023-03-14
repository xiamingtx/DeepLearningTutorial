#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:LinearRegression.py
# author:xm
# datetime:2023/2/11 16:39
# software: PyCharm

"""
使用PyTorch 实现 线性回归

PyTorch Fashion(风格)

1、prepare data

2、design model using Class (inherit from torch.nn.Module)  # 目的是为了前向传播forward，即计算y hat(预测值)

3、Construct loss and optimizer (using PyTorch API) 其中，计算loss是为了进行反向传播，optimizer是为了更新梯度。

4、Training cycle (forward, backward, update)
"""

# import module your need
import torch
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

# 1. prepare dataset
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


# 2. design model
# 继承 torch.nn.Module
class LinearModel(torch.nn.Module):
    # 构造函数
    def __init__(self):
        # 调用父类的构造函数 parameter: (定义的类名, self)
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1)

    # 必须实现forward函数
    def forward(self, x):
        y_pred = self.linear(x)  # linear是torch.nn.Linear的对象 这样使用其实是调用了父类(torch.nn.Module)的__call__函数
        return y_pred


model = LinearModel()

# 3、Construct loss and optimizer
# criterion = torch.nn.MSELoss(size_average=False)  # warning: size_average and reduce args要被弃用了
criterion = torch.nn.MSELoss(reduction='sum')  # define loss function: MSE
# define optimizer: SGD(parameters you wanna optimize, learning rate)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_list = []
mse_list = []

# 4、Training cycle
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())  # 这里会去调用相应类的__str__函数
    epoch_list.append(epoch)
    mse_list.append(loss.item())

    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向传播
    optimizer.step()  # update

# Output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# Test Model
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

# 绘图
plt.plot(epoch_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
