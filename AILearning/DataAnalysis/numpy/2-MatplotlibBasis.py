#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:2-MatplotlibBasis.py
# author:xm
# datetime:2023/2/14 13:28
# software: PyCharm

"""
Matplotlib 基础
在使用Numpy之前，需要了解一些画图的基础。

Matplotlib是一个类似Matlab的工具包，主页地址为

http://matplotlib.org
"""

# 导入 matplotlib 和 numpy：
import numpy as np
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

# plot 二维图
# plt.plot(y)
# plt.plot(x, y)
# plt.plot(x, y, format_string)

# 只给定 y 值，默认以下标为 x 轴：
x = np.linspace(0, 2 * np.pi, 50)
# plt.plot(np.sin(x))
# plt.show()

# 给定 x 和 y 值：
# plt.plot(x, np.sin(x))
# plt.show()

# 多条数据线：
# plt.plot(x, np.sin(x),
#          x, np.sin(2 * x))
# plt.show()

# 使用字符串，给定线条参数：
# plt.plot(x, np.sin(x), 'r-^')
# plt.show()

# 多线条
# 更多参数设置，请查阅帮助。事实上，字符串使用的格式与Matlab相同。
# plt.plot(x, np.sin(x), 'b-o',
#          x, np.sin(2 * x), 'r-^')
# plt.show()

# scatter 散点图
# scatter(x, y)
# scatter(x, y, size)
# scatter(x, y, size, color)

# 假设我们想画二维散点图：
# plt.plot(x, np.sin(x), 'bo')
# plt.show()

# 可以使用 scatter 达到同样的效果：
# 事实上，scatter函数与Matlab的用法相同，还可以指定它的大小，颜色等参数：
# plt.scatter(x, np.sin(x))
# plt.show()

# x = np.random.rand(200)
# y = np.random.rand(200)
# size = np.random.rand(200) * 30
# color = np.random.rand(200)
# plt.scatter(x, y, size, color)
# # 显示颜色条
# plt.colorbar()
# plt.show()

# 使用figure()命令产生新的图像：
t = np.linspace(0, 2 * np.pi, 50)
x = np.sin(t)
y = np.cos(t)
# plt.figure()
# plt.plot(x)
# plt.figure()
# plt.plot(y)
# plt.show()

# 或者使用 subplot 在一幅图中画多幅子图：
# plt.subplot(1, 2, 1)  # 一行两列 第一列
# plt.plot(x)
# plt.subplot(1, 2, 2)  # 一行两列 第二列
# plt.plot(y)
# plt.show()

# 向图中添加数据
# 默认多次 plot 会叠加：
# 可以跟Matlab类似用 hold(False)关掉，这样新图会将原图覆盖：
# plt.plot(x)
# plt.hold(False)
# plt.plot(y)
# plt.hold(True)
# plt.show()
# 现在plt.hold() 似乎已经被删除

# 标签
# 可以在 plot 中加入 label ，使用 legend 加上图例：
# plt.plot(x, label='sin')
# plt.plot(y, label='cos')
# plt.legend()
# plt.show()

# 或者直接在 legend中加入：
# plt.plot(x)
# plt.plot(y)
# plt.legend(['sin', 'cos'])
# plt.show()

# 坐标轴，标题，网格
# 可以设置坐标轴的标签和标题：
# plt.plot(x, np.sin(x))
# plt.xlabel('radians')
# 可以设置字体大小
# plt.ylabel('amplitude', fontsize='large')
# plt.title('Sin(x)')
# plt.show()

# 用 'grid()' 来显示网格：
# plt.plot(x, np.sin(x))
# plt.xlabel('radians')
# plt.ylabel('amplitude', fontsize='large')
# plt.title('Sin(x)')
# plt.grid()
# plt.show()

# 清除、关闭图像
# 清除已有的图像使用：
# plt.clf()
# 关闭当前图像：
# close()
# 关闭所有图像：
# close('all')

# imshow 显示图片
# 灰度图片可以看成二维数组：
# 导入lena图片
# from scipy.misc import lena
# img = lena()
# img
# scipy.misc 已被移除

# 直方图
# 从高斯分布随机生成1000个点得到的直方图：
plt.hist(np.random.randn(1000))
plt.show()

