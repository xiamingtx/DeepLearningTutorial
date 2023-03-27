#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:GradientDescent.py
# author:xm
# datetime:2023/2/10 20:03
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# define learning rate
lr = 0.01

# initial guess of weight
w = 1.0


# define the model linear model y = w*x
def forward(x):
    return x * w


# define the cost function MSE
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# define the gradient function  GD
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= lr * grad_val  # update weight
    print('epoch:', epoch, 'w =', w, 'loss =', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
