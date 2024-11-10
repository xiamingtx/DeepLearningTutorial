#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:net.py
# author:xm
# datetime:2024/7/12 14:35
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
from torch import Tensor, nn
from torch.nn import Module


class AdaLayerNorm(Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: Tensor) -> torch.Tensor:
        x = self.norm(x)

        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2, -1)
        x_embed = x * scale + shift

        return x + x_embed


class ResBlock(Module):
    def __init__(self, dim_in, dim_out, bias=True, n_cond=1000):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)

        self.n_cond = n_cond
        if n_cond > 0:
            self.norm = AdaLayerNorm(n_cond, self.dim_out)
        else:
            self.norm = nn.LayerNorm(dim_out)
        self.activation = nn.SiLU(inplace=True)

        if self.dim_in != self.dim_out:
            self.skip = nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def forward(self, x, t):
        identity = x
        if self.skip is not None:
            identity = self.skip(identity)

        x = self.dense(x)
        norm_inputs = (x, t) if self.n_cond > 0 else (x,)
        x = self.norm(*norm_inputs)

        x += identity
        x = self.activation(x)

        return x


class BasicBlock(Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.dense(x)
        out = self.activation(out)

        return out


class ResMLP(Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, n_cond=1000):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l == 0:
                net.append(BasicBlock(self.dim_in, self.dim_hidden, bias=bias))
            elif l != num_layers - 1:
                net.append(ResBlock(self.dim_hidden, self.dim_hidden,
                                    bias=bias, n_cond=n_cond))
            else:
                net.append(nn.Linear(self.dim_hidden, self.dim_out, bias=bias))
        self.net = nn.ModuleList(net)

    def forward(self, x, t):
        for l in range(self.num_layers):
            net_inputs = (x, t) if l not in (0, self.num_layers - 1) else (x,)
            x = self.net[l](*net_inputs)

        return x
