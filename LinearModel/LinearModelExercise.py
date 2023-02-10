#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:LinearModelExercise.py
# author:xm
# datetime:2023/2/10 10:55
# software: PyCharm

"""
this is function  description 
"""
# import module your need

# Numpy
import numpy
# For plotting
import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

# define training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(w: numpy.ndarray, b: numpy.ndarray, x: float) -> numpy.ndarray:
    return w * x + b  # 矩阵计算


def loss(y_hat: numpy.ndarray, y: float) -> numpy.ndarray:
    return (y_hat - y) ** 2


w_cor = numpy.arange(0.0, 4.0, 0.1)
b_cor = numpy.arange(-2.0, 2.1, 0.1)

# 此处直接使用矩阵进行计算
w, b = numpy.meshgrid(w_cor, b_cor)
mse = numpy.zeros(w.shape)  # 定义误差（矩阵 实质上是每个点(w, b)对应的loss 全0）

for x, y in zip(x_data, y_data):
    _y = forward(w, b, x)  # w 每次都是[0., 0.1, ……, 3.9]   而b依次为[-2, -2, …… -2]、[-1.9, -1.9, ……, -1.9] …… [2, 2, …… 2]
    mse += loss(_y, y)
mse /= len(x_data)  # 均方误差

# 绘图
h = plt.contourf(w, b, mse)

fig = plt.figure()
# ax = Axes3D(fig)
ax = plt.axes(projection='3d')
plt.xlabel(r'w', fontsize=20, color='cyan')
plt.ylabel(r'b', fontsize=20, color='cyan')
ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()

