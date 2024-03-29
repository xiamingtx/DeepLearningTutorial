#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:model.py
# author:xm
# datetime:2024/3/29 22:36
# software: PyCharm

"""
follow the tutorial: https://www.youtube.com/watch?v=oM4VmoabDAI, implement LLaMA2 from scratch in PyTorch
"""

# import module your need
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass  # A decorator that automatically generate magic funcs like __init__() and __repr__()
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # Number of heads for the queries
    n_kv_heads: Optional[int] = None  # Number of heads for the K and V (Grouped Query Attention i.e. GQA is employed)
    vocab_size: int = -1  # This will be set when we load the tokenizer
    multiple_of: int = 256  # dim of FFN
    ffn_dim_multiplier: Optional[float] = None  # GQA decrease params, increasing FFN params to keep model performance
    norm_eps: float = 1e-5  # avoid zero-division

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: Optional[str] = None


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super(Transformer, self).__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, epes=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # RoPE
        self.freqs_complex = precomute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                             self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (bs, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (bs, seq_len) -> (bs, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
