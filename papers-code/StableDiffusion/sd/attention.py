#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:attention.py
# author:xm
# datetime:2024/5/6 11:17
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super(SelfAttention, self).__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (bs, seq_len, dim)
        input_shape = x.shape
        bs, seq_len, d_embed = input_shape
        intermim_shape = (bs, seq_len, self.n_heads, self.d_head)

        # (bs, seq_len, dim) -> (bs, seq_len, 3 * dim) -> 3 tensors of shape (bs, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (bs, seq_len, dim) -> (bs, seq_len, h, dim / h) -> (bs, h, seq_len, dim / h)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (bs, h, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (bs, h, seq_len, seq_len) @ (bs, h, seq_len, dim / h) -> (bs, h, seq_len, dim / h)
        output = weight @ v
        # (bs, h, seq_len, dim / h) -> (bs, seq_len, h, dim / h)
        output = output.transpose(1, 2)
        # (bs, seq_len, h, dim / h) -> (bs, seq_len, dim)
        output = output.reshape(input_shape)
        # (bs, seq_len, dim) -> (bs, seq_len, dim)
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super(CrossAttention, self).__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (latent): (bs, seq_len_q, dim_q)
        # y: (context): (bs, seq_len_kv, dim_kv) = (bs, 77, 768)
        input_shape = x.shape
        bs, seq_len, d_embed = input_shape

        interim_shape = (bs, -1, self.n_heads, self.d_heads)

        # Multiply query by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)

        output = self.out_proj(output)

        return output
