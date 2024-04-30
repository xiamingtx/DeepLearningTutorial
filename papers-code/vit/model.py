#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:model.py
# author:xm
# datetime:2024/3/19 19:36
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # (1, num_patches, embed_dim) -> (bs, num_patches, embed_dim)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # (bs, c, w, h) -> (bs, embed_dim, pw, ph) -> (bs, embed_dim, num_patches) -> (bs, num_patches, embed_dim)
        x = self.patcher(x).permute(0, 2, 1)
        # (bs, num_patches, embed_dim) cat (bs, 1, embed_dim) -> (bs, num_patches + 1, embed_dim)
        x = torch.cat([x, cls_token], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


class Vit(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout,
                 num_heads, activation, num_encoders, num_classes):
        super(Vit, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                   activation=activation, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.mlp = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.mlp(x[:, 0, :])
        return x


if __name__ == '__main__':
    model = Vit(3, 16, 768, 196, 0, 12, nn.GELU(), 6, 1000)
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)
