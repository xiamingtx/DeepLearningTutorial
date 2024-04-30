#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:time_emb.py
# author:xm
# datetime:2024/4/30 22:11
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import math

import torch
import torch.nn as nn
from config import T


class TimeEmbedding(nn.Module):
    def __init__(self, emb_size):
        super(TimeEmbedding, self).__init__()
        self.half_emb_size = emb_size // 2
        half_emb = torch.exp(torch.arange(self.half_emb_size) * (-1 * math.log(10000) / self.half_emb_size - 1))
        self.register_buffer('half_emb', half_emb)

    def forward(self, t):
        t = t.view(t.size(0), -1)
        half_emb = self.half_emb.unsqueeze(0).expand(t.size(0), self.half_emb_size)
        half_emb_t = half_emb * t
        embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1)
        return embs_t


if __name__ == '__main__':
    time_emb = TimeEmbedding(16)
    t = torch.randint(0, T, (2,))  # random sample two timesteps
    embs = time_emb(t)
    print(embs)
