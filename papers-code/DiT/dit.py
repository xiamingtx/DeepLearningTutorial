#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:dit.py
# author:xm
# datetime:2024/4/30 17:39
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch

from config import T
import torch.nn as nn
from time_emb import TimeEmbedding
from dit_block import DiTBlock


class DiT(nn.Module):
    def __init__(self, img_size, patch_size, channel, emb_size, label_num, dit_num, head):
        super(DiT, self).__init__()

        self.patch_size = patch_size
        self.patch_count = img_size // self.patch_size
        self.channel = channel

        # patchify
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel * patch_size ** 2, kernel_size=patch_size,
                              padding=0, stride=patch_size)
        self.patch_emb = nn.Linear(in_features=channel * patch_size ** 2, out_features=emb_size)
        self.patch_pos_emb = nn.Parameter(torch.rand(1, self.patch_count ** 2, emb_size))

        # time emb
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # label emb
        self.label_emb = nn.Embedding(num_embeddings=label_num, embedding_dim=emb_size)

        # DiT Blocks
        self.dits = nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size, head))

        # layer norm
        self.ln = nn.LayerNorm(emb_size)

        # linear back to patch
        self.linear = nn.Linear(emb_size, channel * patch_size ** 2)

    def forward(self, x, t, y):
        """
        :param x: (bs, c, h, w)
        :param t: (bs, )
        :param y: (bs, )
        :return: (bs, c, h, w)
        """
        # label emb
        y_emb = self.label_emb(y)
        # time emb
        t_emb = self.time_emb(t)
        # condition emb
        cond = y_emb + t_emb
        # patch emb
        # patchify
        x = self.conv(x)  # (bs, c * patch_size ** 2, patch_count, patch_count)
        x = x.permute(0, 2, 3, 1)  # (bs, patch_count, patch_count, c * patch_size ** 2)
        x = x.view(x.size(0), self.patch_count ** 2, -1)  # (bs, patch_count ** 2, c * patch_size ** 2)
        # embedding
        x = self.patch_emb(x)  # (bs, patch_count ** 2, emb_size)
        x = x + self.patch_pos_emb  # (bs, patch_count ** 2, emb_size)
        # dit blocks
        for dit in self.dits:
            x = dit(x, cond)
        # layer norm
        x = self.ln(x)  # (bs, patch_count ** 2, emb_size)
        # linear back to patch
        x = self.linear(x)  # (bs, patch_count ** 2, c * patch_size ** 2)
        # reshape
        x = x.view(x.size(0), self.patch_count, self.patch_count, self.channel, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5)  # (bs, c, patch_count, patch_count, patch_size, patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (bs, c, patch_count, patch_size, patch_count, patch_size)
        x = x.reshape(x.size(0), self.channel, self.patch_count * self.patch_size, self.patch_count * self.patch_size)
        return x


if __name__ == '__main__':
    dit = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4)
    x = torch.randn(5, 1, 28, 28)
    t = torch.randint(0, T, (5,))
    y = torch.randint(0, 10, (5,))
    outputs = dit(x, t, y)
    print(outputs.shape)
