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
    multiple_of: int = 256  # Dimension of FFN
    ffn_dim_multiplier: Optional[float] = None  # GQA decrease params, increasing FFN params to keep model performance
    norm_eps: float = 1e-5  # Avoid zero-division

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: Optional[str] = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paper, the dimension of the embedding must be even.
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula: theta_i = 10000 ^ (-2 (i-1) / dim) for i = [1, 2, ..., dim / 2]
    # shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # shape: (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # construct the positions (the "m" parameter)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    # shape: (seq_len) outer_product* (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
    # (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (bs, seq_len, n_heads, head_dim) -> (bs, seq_len, n_heads, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len ,1 , head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (bs, seq_len, n_heads, head_dim / 2) * (1, seq_len ,1 , head_dim / 2) -> (bs, seq_len, n_heads, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (bs, seq_len, n_heads, head_dim / 2) -> (bs, seq_len, n_heads, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (bs, seq_len, n_heads, head_dim / 2, 2) -> (bs, seq_len, n_heads, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (bs, seq_len_kv, n_kv_heads, h_dim) -> (bs, seq_len_kv, n_kv_heads, 1, h_dim)
            x[:, :, :, None, :]
                # (bs, seq_len_kv, n_kv_heads, 1, h_dim) -> (bs, seq_len_kv, n_kv_heads, n_rep, h_dim)
                .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                # (bs, seq_len_kv, n_kv_heads, n_rep, h_dim) -> (bs, seq_len_kv, n_q_heads, h_dim)
                .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (bs, seq_len, dim) * (bs, seq_len, 1) = (bs, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (dim) * (bs, seq_len, dim) = (bs, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super(SelfAttention, self).__init__()
        # Indicates the number of heads for the Key and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the heads of Keys and Values should be repeated to match the head of the Queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (bs, 1, dim)
        # Apply the Wq, Wk and Wv matrices to queries, keys, values
        # (bs, 1, dim) -> (bs, 1, h_q * head_dim)
        xq = self.wq(x)
        # (bs, 1, dim) -> (bs, 1, h_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (b, 1, h_q * head_dim) -> (b, 1, h_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (b, 1, h_kv * head_dim) -> (b, 1, h_kv * head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to queries and keys, does not change the shape of the tensors
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # Retrieve all the cached keys and values so far
        # (bs, seq_len_kv, h_kv, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]

        # Repeat the heads of the K and V to reach the number of heads of the queries
        # (bs, seq_len_kv, h_kv, head_dim) -> (bs, seq_len_kv, h_q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (bs, 1, h_q, head_dim) -> (bs, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        # (bs, seq_len_kv, h_kv, head_dim) -> (bs, h_q, seq_len_kv, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (bs, h_q, 1, head_dim) @ (bs, h_q, head_dim, seq_len_kv) -> (bs, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (bs, h_q, 1, seq_len_kv) @ (bs, h_q, seq_len_kv, head_dim) -> (bs, h_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (bs, h_q, 1, head_dim) -> (bs, 1, h_q, head_dim) -> (bs, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        # (bs, 1, dim) -> (bs, 1, dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super(FeedForward, self).__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super(EncoderBlock, self).__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (bs, seq_len, dim) + (bs, seq_len, dim) -> (bs, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


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

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # RoPE
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
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
