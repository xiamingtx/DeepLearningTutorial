#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:train.py
# author:xm
# datetime:2024/3/18 20:40
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
from dataset import MNIST
from clip import CLIP
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MNIST()

model = CLIP().to(DEVICE)

try:
    model.load_state_dict(torch.load('model.pth'))
except:
    pass

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

"""
    Training Args
"""
ITER_BATCH_COUNT = 10000
BATCH_SIZE = 64
TARGET_COUNT = 10
NUM_WORKERS = 0

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

if __name__ == '__main__':
    for i in range(ITER_BATCH_COUNT):
        while True:
            imgs, labels = next(iter(dataloader))
            if torch.unique(labels).shape[0] < TARGET_COUNT:  # 未覆盖10种数字
                continue
            # 挑选出10个数字
            target = set()
            indices = []
            for j in range(BATCH_SIZE):
                if labels[j].item() in target:
                    continue
                target.add(labels[j].item())
                indices.append(j)
                if len(target) == TARGET_COUNT:
                    break
            imgs = imgs[indices]
            labels = labels[indices]
            break

        logits = model(imgs.to(DEVICE), labels.to(DEVICE))
        targets = torch.arange(0, TARGET_COUNT).to(DEVICE)
        loss_i = criterion(logits, targets)
        loss_t = criterion(logits.permute(1, 0), targets)
        loss = (loss_i + loss_t) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f'iter: {i}, loss: {loss.item()}')
            torch.save(model.state_dict(), 'model.pth')
