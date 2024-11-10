#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:conformal_prediction.py
# author:xm
# datetime:2024/11/9 22:35
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import numpy as np

# 模拟简单数据: 假设有一个3分类问题，包含5个样本
# Softmax输出模拟（模型预测的概率）
softmax_outputs = np.array([
    [0.6, 0.3, 0.1],  # 样本1
    [0.2, 0.5, 0.3],  # 样本2
    [0.4, 0.4, 0.2],  # 样本3
    [0.1, 0.7, 0.2],  # 样本4
    [0.3, 0.4, 0.3]  # 样本5
])

# 校准集真实标签 (假设已知)
cal_labels = np.array([0, 1, 0, 1, 2])

# 第一步：计算非一致性分数 (Nonconformity Score)
# 计算非一致性分数 (1 - softmax对真实类别的概率)
n = len(cal_labels)  # 样本数目
nonconformity_scores = 1 - softmax_outputs[np.arange(n), cal_labels]

# 第二步：确定置信阈值 (Quantile Level)
alpha = 0.05  # 95% 置信水平

# 使用保序预测的分位数方法计算阈值 q_hat
q_hat = np.quantile(nonconformity_scores, 1 - alpha, method='higher')

# 打印非一致性分数和阈值
print("非一致性分数:", nonconformity_scores)
print("置信阈值 q_hat:", q_hat)


# 第三步：生成预测集 (Prediction Sets)
# 使用计算出的 q_hat 生成预测集
def generate_prediction_set(softmax_outputs, q_hat):
    prediction_sets = []
    for pred_probs in softmax_outputs:
        # 选择 softmax 分数大于等于 1 - q_hat 的类别
        prediction_set = np.where(pred_probs >= 1 - q_hat)[0]
        prediction_sets.append(prediction_set)
    return prediction_sets


# 测试集的softmax输出
test_softmax_outputs = np.array([
    [0.5, 0.4, 0.1],  # 样本6
    [0.3, 0.5, 0.2],  # 样本7
    [0.05, 0.05, 0.9],  # 样本8
])

# 获取预测集
prediction_sets = generate_prediction_set(test_softmax_outputs, q_hat)

# 打印结果: [array([0, 1], dtype=int64), array([1], dtype=int64), array([2], dtype=int64)]
print("预测集:", prediction_sets)
