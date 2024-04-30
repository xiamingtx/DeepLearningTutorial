#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:inference.py
# author:xm
# datetime:2024/3/18 21:21
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from dataset import MNIST
import matplotlib.pyplot as plt
import torch
from clip import CLIP
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MNIST()

model = CLIP().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))

if __name__ == '__main__':
    # 1. 对图片分类
    model.eval()
    image, label = dataset[33]
    print('正确分类: ', label)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    targets = torch.arange(0, 10)
    logits = model(image.unsqueeze(0).to(DEVICE), targets.to(DEVICE))
    print(logits)
    print('CLIP分类: ', logits.argmax(-1).item())

    # 2. 图像相似度
    other_images = []
    other_labels = []
    for i in range(1, 101):
        other_image, other_label = dataset[i]
        other_images.append(other_image)
        other_labels.append(other_label)

    # 其他100张图片的向量
    other_img_embs = model.img_enc(torch.stack(other_images, dim=0).to(DEVICE))

    # 当前图片的向量
    img_emb = model.img_enc(image.unsqueeze(0).to(DEVICE))

    # 计算当前图片与100张其他图片的相似度
    logits = img_emb @ other_img_embs.T
    values, indices = logits[0].topk(5)  # 5个最相似的

    plt.figure(figsize=(15, 15))
    for i, img_idx in enumerate(indices):
        plt.subplot(1, 5, i + 1)
        plt.imshow(other_images[img_idx].permute(1, 2, 0))
        plt.title(other_labels[img_idx])
        plt.axis('off')
    plt.show()
