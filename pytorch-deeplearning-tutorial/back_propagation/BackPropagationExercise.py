#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:BackPropagationExercise.py
# author:xm
# datetime:2023/2/11 10:54
# software: PyCharm

"""
model: y = w1 * x ** 2 + w2 * x + b
loss: (y_hat - y) ** 2
"""

# import module your need
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])  # 初始权值
w1.requires_grad = True  # 计算梯度，默认是不计算的
w2 = torch.Tensor([1.0])
w2.requires_grad = True
b = torch.Tensor([1.0])
b.requires_grad = True


def forward(x):
    return w1 * x ** 2 + w2 * x + b


def loss(x, y):  # 构建计算图
    y_pred = forward(x)
    return (y_pred - y) ** 2


print('Predict(before training)', 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data  # 注意这里的grad是一个tensor，所以要取他的data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data
        w1.grad.data.zero_()  # 释放之前计算的梯度
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('Epoch:', epoch, l.item())

print('Predict(after training)', 4, forward(4).item())
