#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:text_encoder.py
# author:xm
# datetime:2024/3/18 20:22
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from torch import nn
import torch


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.emb = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.dense1 = nn.Linear(in_features=16, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=16)
        self.fc = nn.Linear(in_features=16, out_features=8)
        self.ln = nn.LayerNorm(8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.emb(x)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        out = self.ln(self.fc(x))
        return out


if __name__ == '__main__':
    text_encoder = TextEncoder()
    input = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    out = text_encoder(input)
    print(out.shape)
