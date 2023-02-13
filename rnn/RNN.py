#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:RNN.py
# author:xm
# datetime:2023/2/13 20:47
# software: PyCharm

"""
how to use RNN
"""

# import module your need
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# (seqLen, batchSize, inputSize)
inputs = torch.randn(seq_len, batch_size, input_size)  # input of shape(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)  # hidden of shape(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)

print('Output size:', out.shape)        # output of shape(seq_len, batch_size, hidden_size)  torch.Size([3, 1, 2])
print('Output:', out)
print('Hidden size:', hidden.shape)     # hidden of shape(num_layers, batch_size, hidden_size)  torch.Size([1, 1, 2])
print('Hidden:', hidden)
