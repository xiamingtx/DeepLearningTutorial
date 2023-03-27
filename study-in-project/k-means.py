#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:k-means.py
# author:xm
# datetime:2023/3/8 16:52
# software: PyCharm

"""
K-means聚类算法的流程：
1.随机选取K个中心点

2.遍历数据集里面的每个点，看距离哪个中心点最近就分为哪一类，遍历完一共K类

3.把属于一类的点取平均值，得到的平均值作为新的中心点

4.然后不断重复步骤2、3, 直到达到结束条件为止。（当中心点不再变动或变动很小，当达到最大迭代次数）
"""

# import module your need
import numpy as np
import random


def cal_distance(node, center):
    """

    Parameters
    ----------
    node        遍历所有点时得到的点
    center      选出的中心

    Returns     返回点与中心的距离
    -------

    """
    return np.sqrt(np.sum(np.square(node - center)))


def random_center(data, k):
    """

    Parameters
    ----------
    data    数据集
    k       选取k个中心

    Returns     从总的数据集中随机抽样得到k个中心
    -------

    """
    data = list(data)
    return random.sample(data, k)


def get_cluster(data, center):
    """

    Parameters
    ----------
    data        数据集
    center      选出的中心

    Returns     dict{每个中心（类）: array(属于该类的点)}
    -------

    """
    cluster_dict = dict()
    k = len(center)
    for node in data:
        cluster_class = -1
        min_distance = float('inf')
        for i in range(k):
            dist = cal_distance(node, center[i])
            if dist < min_distance:
                cluster_class = i
                min_distance = dist
        if cluster_class not in cluster_dict.keys():
            cluster_dict[cluster_class] = []
        cluster_dict[cluster_class].append(node)
    return cluster_dict


def get_center(cluster_dict, k):
    """

    Parameters
    ----------
    cluster_dict    get_cluster()函数中获得的字典{中心：属于该类的点集}
    k               选出的中心个数

    Returns         根据每一类点的平均值
    -------

    """
    new_center = []
    for i in range(k):
        # 计算每类中点的平均值作为新的center
        center = np.mean(cluster_dict[i], axis=0)
        new_center.append(center)
    return new_center


def cal_variance(cluster_dict, center):
    """

    Parameters
    ----------
    cluster_dict
    center

    Returns     计算出偏差
    -------

    """
    vsum = 0
    # 遍历所有中心
    for i in range(len(center)):
        # 取出所有属于该中心类别的点
        cluster = cluster_dict[i]
        for j in cluster:
            # 遍历所有该类别的点 计算与中心的距离
            vsum += cal_distance(j, center[i])
    return vsum


def k_means(data, k):
    # 从点集中随机抽取k个作为中心点
    center = random_center(data, k)
    print(center)
    cluster_dict = get_cluster(data, center)
    new_variance = cal_variance(cluster_dict, center)
    old_variance = 1
    # 只要偏差 > 0.1 就循环下述步骤
    while abs(old_variance - new_variance) > 0.1:
        # 对之前分类得到的k个类中 每类计算平均得到新的center
        center = get_center(cluster_dict, k)
        cluster_dict = get_cluster(data, center)
        old_variance = new_variance
        new_variance = cal_variance(cluster_dict, center)
    return cluster_dict, center


data = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 1], [9, 8, 7], [7, 8, 9], [8, 9, 7]])
a, b = k_means(data, 2)
print(a, b)
