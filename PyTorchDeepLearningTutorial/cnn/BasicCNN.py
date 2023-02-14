#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:BasicCNN.py
# author:xm
# datetime:2023/2/13 8:05
# software: PyCharm

"""
CNN:  Convolutional Neural Networks 卷积神经网络
inputs -> Feature Extraction(Convolution & Subsampling) -> Fully Connected -> outputs

Example:
import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)
conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=(kernel_size, kernel_size))  # 如果是方阵 也可以只传入一个int

output = conv_layer(input)
print(input.shape)  # torch.Size([1, 5, 100, 100])
print(output.shape)  # torch.Size([1, 10, 98, 98])
print(conv_layer.weight.shape)  # torch.Size([10, 5, 3, 3])

padding & stride Example:
import torch

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
input = torch.Tensor(input).view(1, 1, 5, 5)  # B C W H

conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)  # O I W H
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output.shape)  # torch.Size([1, 1, 5, 5])
"""

# import module your need
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)
        batch_size = x.size(0)
        # (batch, 1, 28, 28) -> (batch, 10, 24, 24) -> (batch, 10, 12, 12)
        x = F.relu(self.pooling(self.conv1(x)))
        # (batch, 10, 12, 12) -> (batch, 20, 8, 8) -> (batch, 20, 4, 4)
        x = F.relu(self.pooling(self.conv2(x)))
        # (batch, 20, 4, 4) -> (batch, 320)
        x = x.view(batch_size, -1)  # flatten
        # (batch, 320) -> (batch, 10)
        x = self.fc(x)
        return x


model = Net()

# 使用GPU
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
        inputs, target = inputs.to(device), target.to(device)  # 迁移到device上 记住 数据和模型要放在同一块显卡上
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
            images, labels = images.to(device), labels.to(device)  # 迁移到device上 记住 数据和模型要放在同一块显卡上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))
    return correct / total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
