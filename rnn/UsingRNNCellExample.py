#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:UsingRNNCellExample.py
# author:xm
# datetime:2023/2/13 21:00
# software: PyCharm

"""
Train a model to learn:
    "hello" -> "ohlol"

The inputs of RNN Cell should be vectors of numbers
we can use dictionary to convert input to indices

h            character index                1                   0 1 0 0
e               e       0                   0                   1 0 0 0
l       ->      h       1         ->        2          ->       0 0 1 0
l               l       2                   2                   0 0 1 0
o               o       3                   3                   0 0 0 1
[Input]         [Dictionary]            [indices]          [One-Hot Vectors]

inputSize = 4
这实际上是一个分类问题
outputSize = 4
"""

# import module your need
import torch

input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e', 'h', 'l', 'o']  # The dictionary
x_data = [1, 0, 2, 2, 3]  # The input sequence is 'hello'
y_data = [3, 1, 2, 3, 2]  # The output sequence is 'ohlol'

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # Convert indices into one-hot vector

# Reshape the inputs to (seqLen, batchSize, inputSize)  torch.Size([5, 1, 4])
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)  # Reshape the labels to (seqLen, 1)  torch.Size([5, 1])


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)  # Provide initial hidden


net = Model(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()

    print('Predicted string: ', end='')
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))



