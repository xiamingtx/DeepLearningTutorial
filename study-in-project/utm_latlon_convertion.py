#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:utm_latlon_convertion.py
# author:xm
# datetime:2023/3/12 13:28
# software: PyCharm

"""
1. utm坐标系与(latitude, longtitude)坐标系的转换
2. 距离的计算
"""

# import module your need
import utm
import numpy as np

R = 6371.393
Pi = np.pi

s_latitude, s_longitude = 40.086133262333, 116.30104133683

s_x, s_y, s_altitude = 440411.8745000544, 4437551.29303103, 34.8353

print('convert latitude & longitude to utm', utm.from_latlon(s_latitude, s_longitude))

print('convert latitude & longitude from utm', utm.to_latlon(s_x, s_y, 50, 'T'))

d_latitude, d_longitude = 40.086208388333, 116.30099986583

d_x, d_y, d_altitude = 440408.4044643425, 4437559.659289575, 33.0813

print('convert latitude & longitude to utm', utm.from_latlon(d_latitude, d_longitude))

print('convert latitude & longitude from utm', utm.to_latlon(d_x, d_y, 50, 'T'))


def calc_distance_utm(source, destination):
    """

    Parameters
    ----------
    source      utm坐标系下起始位置坐标
    destination     utm坐标系下目标位置坐标

    Returns  返回utm坐标系下的距离
    -------

    """
    return np.sqrt(np.sum(np.square(source - destination)))


def calc_distance_latlon(source, destination):
    """

    Parameters
    ----------
    source          起始的经纬度
    destination     目标的经纬度

    Returns         距离（米）
    -------

    """
    # source
    source_latitude, source_longitude = source
    # 转为空间直角坐标
    source_x = np.cos(np.radians(source_latitude)) * np.cos(np.radians(source_longitude))
    source_y = np.cos(np.radians(source_latitude)) * np.sin(np.radians(source_longitude))
    source_z = np.sin(np.radians(source_latitude))
    source = np.array((source_x, source_y, source_z))

    # destination
    destination_latitude, destination_longitude = destination
    # 转为空间直角坐标
    destination_x = np.cos(np.radians(destination_latitude)) * np.cos(np.radians(destination_longitude))
    destination_y = np.cos(np.radians(destination_latitude)) * np.sin(np.radians(destination_longitude))
    destination_z = np.sin(np.radians(destination_latitude))
    destination = np.array((destination_x, destination_y, destination_z))

    # 开始计算
    cos_alpha = sum(source * destination) / (sum(np.square(source)) * sum(np.square(destination))) ** 0.5

    alpha = np.arccos(cos_alpha)

    L = alpha * R * 1000
    return L


print(calc_distance_utm(np.array((s_x, s_y, s_altitude)), np.array((d_x, d_y, d_altitude))))
print(calc_distance_latlon(np.array((s_latitude, s_longitude)), np.array((d_latitude, d_longitude))))
