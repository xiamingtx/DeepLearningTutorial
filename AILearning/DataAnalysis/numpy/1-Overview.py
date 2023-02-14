#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:1-Overview.py
# author:xm
# datetime:2023/2/14 12:57
# software: PyCharm

"""
导入numpy
Numpy是Python的一个很重要的第三方库，很多其他科学计算的第三方库都是以Numpy为基础建立的。

Numpy的一个重要特性是它的数组计算。

"""

# 在使用Numpy之前，我们需要导入numpy包：
import numpy as np
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

# 假如我们想将列表中的每个元素增加1，但列表不支持这样的操作（报错）:
# a = [1, 2, 3, 4]
# print(a + 1)
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# <ipython-input-3-068856d2a224> in <module>()
#  1 a = [1, 2, 3, 4]
# ----> 2  a + 1
#
# TypeError: can only concatenate list (not "int") to list

a = [1, 2, 3, 4]
a = np.array(a)
print('a + 1', a + 1)
# [2 3 4 5]  array 数组支持每个元素加 1 这样的操作：

# 与另一个 array 相加，得到对应元素相加的结果：
b = np.array([2, 3, 4, 5])
print('a + b', a + b)

# 对应元素相乘：
print('a * b', a * b)

# 对应元素乘方：
print('a ** b', a ** b)

# 提取数组元素
# 提取第一个元素：
print('a[0]', a[0])

# 提取前两个元素：
print('a[:2]', a[:2])

# 最后两个元素：
print('a[-2:]', a[-2:])

# 将它们相加：
print('a[:2] + a[-2:]', a[:2] + a[-2:])

# 修改数组形状
# 查看 array 的形状：
print('a.shape', a.shape)

# 修改 array 的形状：
a.shape = 2, 2
print('a:')
print(a)


# 多维数组
# a 现在变成了一个二维的数组，可以进行加法：
print('a + a')
print(a + a)

# 乘法仍然是对应元素的乘积，并不是按照矩阵乘法来计算：
print('a * a')
print(a * a)

# 画图
# linspace 用来生成一组等间隔的数据：
a = np.linspace(0, 2 * np.pi, 21)
# %precision 3  原文
np.set_printoptions(precision=3)  # 设置精度
print('linspace exercise:')
print(a)

# 三角函数
b = np.sin(a)
print('sin(a): ')
print(b)

# plt.plot(a, b)
# plt.show()  # 要显示图像的话打开

# 从数组中选择元素
# 假设我们想选取数组b中所有非负的部分，首先可以利用 b 产生一组布尔值：
print('b >= 0')
print(b >= 0)

mask = b >= 0

# 'r'对应的是颜色是 红色，'o'对应的标记说明是 实心圈标记
plt.plot(a[mask], b[mask], 'ro')

plt.show()

