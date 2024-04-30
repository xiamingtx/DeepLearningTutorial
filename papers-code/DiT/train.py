#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:train.py
# author:xm
# datetime:2024/5/1 0:49
# software: PyCharm

"""
follow tutorial: https://www.bilibili.com/video/BV13K421h79z/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=e472d54fbaf4a2a11e9526662ac3a29b
"""

# import module your need
from config import T
from torch.utils.data import DataLoader
from dataset import MNIST
from diffusion import forward_add_noise
import torch
from torch import nn
import os
from dit import DiT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 500
BATCH_SIZE = 1

if __name__ == '__main__':
    dataset = MNIST()

    model = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4).to(DEVICE)  # 模型

    try:
        model.load_state_dict(torch.load('model.pth'))
    except:
        pass

    optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = nn.L1Loss()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                            persistent_workers=True)

    model.train()

    iter_count = 0
    for epoch in range(EPOCH):
        for imgs, labels in dataloader:
            x = imgs * 2 - 1  # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应
            t = torch.randint(0, T, (imgs.size(0),))
            y = labels

            x, noise = forward_add_noise(x, t)
            pred_noise = model(x.to(DEVICE), t.to(DEVICE), y.to(DEVICE))

            loss = loss_fn(pred_noise, noise.to(DEVICE))

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            if iter_count % 1000 == 0:
                print('epoch:{} iter:{},loss:{}'.format(epoch, iter_count, loss))
                torch.save(model.state_dict(), '.model.pth')
                os.replace('.model.pth', 'model.pth')
            iter_count += 1
