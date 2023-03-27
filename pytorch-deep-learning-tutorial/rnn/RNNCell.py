#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:RNNCell.py
# author:xm
# datetime:2023/2/13 19:53
# software: PyCharm

"""
RNN 的本质是一个线性层
"""

# import module your need
import torch

batch_size = 1  # 批处理大小
seq_len = 3  # 序列长度
input_size = 4  # 输入维度
hidden_size = 2  # 隐层(输出)维度

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# (seqLen 序列长度, batchSize, inputSize 特征数)
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

# 这个循环就是处理seq_len长度的数据
for idx, data in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('Input size:', data.shape, data)  # input.shape = (batchSize, inputSize)  torch.Size([1, 4])

    hidden = cell(data, hidden)

    print('hidden size:', hidden.shape, hidden)  # output.shape = (batchSize, hiddenSize)  torch.Size([1, 2])
    print(hidden)
