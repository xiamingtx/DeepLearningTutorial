#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:ResNets.py
# author:xm
# datetime:2023/2/13 19:28
# software: PyCharm

"""
如果 网络很深 可能会出现梯度消失(每一层的梯度都很小 经过链式法则相乘 就会导致趋于0) 那么前面的层权重就得不到更新
我们可以通过逐层训练来解决 但是这在深度学习中不适用
可以使用跳连接，H(x) = F(x) + x,张量维度必须一样，加完后再激活。不要做pooling，张量的维度会发生变化。
H'(x) = F'(x) + 1 这样当F'(x) -> 0时， H'(x) -> 1 而不是趋于0 不会导致梯度消失

这里我们介绍 残差网络(ResNets) 以及其组成部分: 残差块(ResidualBlock)

homework1:
read paper: He K, Zhang X, Ren S, et al. Identity Mappings in Deep Residual Networks
https://arxiv.org/abs/1603.05027

homework2:
read paper: Huang G, Liu Z, Laurens V D M, et al. Densely Connected Convolutional Networks
https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf

接下来的路怎么走:
1. 理论 《深度学习》 花书 从数学和工程学的角度重新理解深度学习的理念
2. 阅读 PyTorch文档(通读一遍)
3. 变现经典工作(找经典深度学习论文 进行复现)
    3.1 读代码
    3.2 尝试自己写 复现
    3.3 过程1、2不断循环 这样才是一个学习的过程
4. 扩充视野(先具备前三点能力)
    4.1 选定特定的领域 大量阅读论文 想创新点
    4.2 解决知识上的盲点
    4.3 锻炼代码能力
    4.4 转化成果
"""

# import module your need
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='../../data/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../../data/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # 88 = 24x3 + 16

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)

        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # 迁移到GPU上
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 迁移到GPU上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("accuracy on test set:%d %% [%d/%d]" % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
