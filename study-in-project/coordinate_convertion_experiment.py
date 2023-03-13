#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:coordinate_convertion_experiment.py
# author:xm
# datetime:2023/3/13 16:43
# software: PyCharm

"""
las与utm误差修正实验
"""

# import module your need
import utm
import numpy as np

origin = np.array([0, 0, 0])  # 原点
origin_las_coordinate = np.array([440411.8745000544, 4437551.29303103, 34.8353])  # 点云坐标中拿到的x, y, z
start_lat_lon_alt = (40.086133262333, 116.30104133683, 34.8353)  # 设定原点处的latitude、longitude、altitude
start_utm = np.append(np.array(utm.from_latlon(*start_lat_lon_alt[:2])[:2]), start_lat_lon_alt[-1])

delta = origin_las_coordinate - start_utm  # 起始las坐标和utm的误差

current_lat_lon_alt = (40.086208388333, 116.30099986583, 33.0813)  # 当前位置处的latitude与longitude
current_utm = np.append(np.array(utm.from_latlon(*current_lat_lon_alt[:2])[:2]), current_lat_lon_alt[-1])
current_relative_utm = current_utm - start_utm + delta

print(start_utm)
print(current_utm)
print(current_relative_utm)

print('distance: ', np.sqrt(np.sum(np.square(current_relative_utm - origin))))
