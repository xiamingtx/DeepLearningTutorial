#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:StochasticGradientDescent.py
# author:xm
# datetime:2023/2/10 20:19
# software: PyCharm

"""
随机梯度下降法在神经网络中被证明是有效的。效率较低(时间复杂度较高)，学习性能较好。
随机梯度下降法和梯度下降法的主要区别在于：

1、损失函数由cost()更改为loss()。cost是计算所有训练数据的损失，loss是计算一个训练数据的损失。对应于源代码则是少了两个for循环。

2、梯度函数gradient()由计算所有训练数据的梯度更改为计算一个训练数据的梯度。

3、本算法中的随机梯度主要是指，每次拿一个训练数据来训练，然后更新梯度参数。本算法中梯度总共更新100(epoch)x3 = 300次。
梯度下降法中梯度总共更新100(epoch)次。

如果使用GD 在鞍点处 我们如果计算cost（统计所有样本） 就可能没法再进行梯度下降了。
但是如果使用SGD 因为我们的数据通常会有噪声, 通过随机取其中一个样本的loss进行梯度下降 我们的算法可能可以继续进行
这使得我们在更新时 有可能跨越过鞍点 向最优值前进 且 SGD在神经网络中验证非常有效

梯度下降可以并行 性能低 运算效率（时间复杂度）高
随机梯度下降无法并行 性能高 运算效率（时间复杂度）低
所以在实践中 我们通常使用Batch梯度下降（当batch=1时即为随机梯度下降） 更正式的, 我们称其为Mini-Batch
"""

# import module your need
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


# calculate loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# define the gradient function  sgd
def gradient(x, y):
    return 2 * x * (x * w - y)


epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad  # update weight by every grad of sample of training set
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
    print("progress:", epoch, "w=", w, "loss=", l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
