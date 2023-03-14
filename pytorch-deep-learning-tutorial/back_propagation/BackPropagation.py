#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:BackPropagation.py
# author:xm
# datetime:2023/2/11 9:24
# software: PyCharm

"""
1.  对于比较简单的模型 如单层的线性模型 我们可以直接通过解析式来求解梯度 实现梯度下降
    但是在深度网络模型中, 网络的权重参数太多 使用解析式的方式太过复杂不可行
    因此我们采用反向传播(BackPropagation)

2.  反向传播是通过求导的链式法则来实现的
    对于求导公式不太了解的同学 可以查看 matrix-cook-book
    网址: http://faculty.bicmr.pku.edu.cn/~wenzw/bigdata/matrix-cook-book.pdf

3.  这边咱们再补充一下 为什么要引入激活函数？ -> 为了引入非线性
    如果使用多层的线性模型 y = U(VX + c) + d = UVX + Uc + d = WX + b
    本质上它还是一个线性模型……  为此 我们可以引入如sigmoid、softmax、ReLU等函数
"""

# import module your need
import torch

# define training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

W = torch.Tensor([1.0])  # define w
W.requires_grad = True  # 表明w需要进行梯度计算


# define linear model  y = x * w
def forward(x):
    return x * W


# define loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, W.grad.item())  # W.grad.item() 可以取得梯度的标量
        W.data = W.data - 0.01 * W.grad.data

        W.grad.data.zero_()

    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())
