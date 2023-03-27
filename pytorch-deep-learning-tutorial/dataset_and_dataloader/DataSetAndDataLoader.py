#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:dataset_and_dataloader.py
# author:xm
# datetime:2023/2/12 17:29
# software: PyCharm

"""
我们学习了梯度下降和随机梯度下降
在梯度下降中 我们通过计算所有样本来获取loss
在随机梯度下降中 我们随机获取单个样本来计算loss 这可以帮助我们克服鞍点 但是它的缺点是计算时间会过长
所以在深度学习中 我们使用 Mini-Batch 在性能和时间复杂度中进行折中

1、需要mini_batch 就需要import DataSet和DataLoader

2、继承DataSet的类需要重写init，getitem,len魔法函数。分别是为了加载数据集，获取数据索引，获取数据总量。

3、DataLoader对数据集先打乱(shuffle)，然后划分成mini_batch。

4、len函数的返回值 除以 batch_size 的结果就是每一轮epoch中需要迭代的次数。

5、inputs, labels = data中的inputs的shape是[32,8],labels 的shape是[32,1]。也就是说mini_batch在这个地方体现的

6、diabetes.csv数据集老师给了下载地址，该数据集需和源代码放在同一个文件夹内。
"""

# import module your need
import torch
import numpy as np
# Dataset is an abstract class. We can define our class inherited from this class.
from torch.utils.data import Dataset
# DataLoader is a class to help us loading dataset in PyTorch.
from torch.utils.data import DataLoader


# prepare dataset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列) 这里会返回有多少条数据
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('../data/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # num_workers 多线程


# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle:  forward, backward, update
if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            # 1. prepare data
            inputs, labels = data
            # 2. Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 3. Backward
            optimizer.zero_grad()
            loss.backward()
            # 4. Update
            optimizer.step()
