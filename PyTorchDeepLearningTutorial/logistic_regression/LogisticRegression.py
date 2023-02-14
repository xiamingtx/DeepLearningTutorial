#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:LogisticRegression.py
# author:xm
# datetime:2023/2/12 10:16
# software: PyCharm

"""
sigmoid函数  f(x) = 1 / (1 + e^(-x)) = (e^x) / (1 + e^x)
当x趋于负无穷 f(x)趋于0  当 x = 0  f(x) = 0.5  当x趋于正无穷 f(x)趋于1
input: x -> (wx + b) -> output: sigmoid(wx + b)
通过sigmoid函数 将结果转换到0~1之间 相当于输出概率

二分类 loss = -(y * log(y_hat) + (1 - y) * log(1 - y_hat))  我们称之为 Binary CrossEntropy Loss (BCE)
来源于交叉熵 CrossEntropy: 有P1(x)、P2(x)  CrossEntropy = Σ P1(xi)ln(P2(xi))
"""

# import module your need
import torch
# import torch.nn.functional as F
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 视频中代码F.sigmoid(self.linear(x))会引发warning，此处更改为torch.sigmoid(self.linear(x))
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
# 实质上 如果求平均 求出的梯度前面会带一个 1 / N 影响学习率的调节 求不求平均都可以hh
# criterion = torch.nn.BCELoss(size_average=False)
criterion = torch.nn.BCELoss(reduction='sum')  # reduction 可以取值 mean sum None
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 可以尝试自己调节lr   0.01 -> 0.03 -> 0.1 ……

epoch_list = []
bce_list = []
# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    bce_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

# 绘图
plt.plot(epoch_list, bce_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
