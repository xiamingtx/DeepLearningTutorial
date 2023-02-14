#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:KaggleOttoExercise.py
# author:xm
# datetime:2023/2/12 23:00
# software: PyCharm

"""
Otto Group Product Classification Challenge
https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview
"""

# import module your need
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning


# 函数将字符型标签转换为数值标签,方便计算交叉熵
def labels2id(labels):
    target_id = []
    target_labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    for label in labels:
        target_id.append(target_labels.index(label))
    return target_id


# 定义数据集类
class TrainDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        labels = data['target']
        self.len = data.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.tensor(np.array(data)[:, 1:-1].astype(float))
        self.y_data = labels2id(labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_dataset = TrainDataset('../dataset/otto/train.csv')
# 数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(93, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.linear4 = torch.nn.Linear(16, 9)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.linear4(x)
        return x


model = Model()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)

loss_list = []


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:  # 每300轮打印一次结果
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


# 开始训练
if __name__ == '__main__':
    for epoch in range(50):
        train(epoch)


# 预测保存函数，用于保存预测结果
def predict_save():
    with torch.no_grad():
        test_data = pd.read_csv('../dataset/otto/test.csv')
        x_text = torch.tensor(np.array(test_data)[:, 1:].astype(float))
        y_pred = model(x_text.float())
        _, predicted = torch.max(y_pred, dim=1)  # 这里先取出最大概率的索引，即是所预测的类别。
        out = pd.get_dummies(predicted)  # get_dummies 利用pandas实现one hot encode，方便保存为预测文件。

        labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
        # 添加列标签
        out.columns = labels
        # 插入id行
        out.insert(0, 'id', test_data['id'])
        result = pd.DataFrame(out)
        result.to_csv('my_predict.csv', index=False)

    # 画损失函数曲线
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()


predict_save()
