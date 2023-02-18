#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:SoftmaxClassifier.py
# author:xm
# datetime:2023/2/12 19:47
# software: PyCharm

"""
多分类问题 本质上 我们输出的是一个分布
softmax: 输出P(y = i) = (e^Zi) / (Σ e^Zj) 保证所有值都在0~1之间 且所有输出的和为1
NLLLoss: Negative Log Likelihood Loss
Loss(y_hat, y) = -y * log(y_hat)
附上一篇关于NLLLoss与CrossEntropyLoss差异的文章 https://blog.csdn.net/cnhwl/article/details/125518586

1、softmax的输入不需要再做非线性变换，也就是说softmax之前不再需要激活函数(relu)。
softmax两个作用，如果在进行softmax前的input有负数，通过指数变换，得到正数。所有类的概率求和为1。

2、y的标签编码方式是one-hot。我对one-hot的理解是只有一位是1，其他位为0。(但是标签的one-hot编码是算法完成的，算法的输入仍为原始标签)

3、多分类问题，标签y的类型是LongTensor。比如说0-9分类问题，如果y = torch.LongTensor([3])，对应的one-hot是[0,0,0,1,0,0,0,0,0,0].
(这里要注意，如果使用了one-hot，标签y的类型是LongTensor，糖尿病数据集中的target的类型是FloatTensor)

4、CrossEntropyLoss <==> LogSoftmax + NLLLoss。也就是说使用CrossEntropyLoss最后一层(线性层)是不需要做其他变化的；
使用NLLLoss之前，需要对最后一层(线性层)先进行SoftMax处理，再进行log操作。

实现案例:
1.
import numpy as np
y = np.array ([1, 0, 0])
z = np.array ([0.2, 0.1,-0.1])
y _pred = np.exp(z) / np.exp(z).sum()
loss = (- y *np.log(y_pred)).sum()
print (loss)

2.
import torch
y = torch.LongTensor([0])
z = torch.Tensor([[0.2, 0.1, -0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z, y)
print(loss)

import torch

3.
criterion = torch.nn.CrossEntropyLoss()
Y = torch.LongTensor([2, 0, 1])
Y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]])
Y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]])
l1 = criterion(Y_pred1, Y)
l2 = criterion(Y_pred2, Y)
# Batch Loss1 = tensor(0.4966)
# Batch Loss2 = tensor(1.2389)
print("Batch Loss1 = ", l1.data, "\nBatch Loss2=", l2.data)
"""

# import module your need
import torch
# For Constructing DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# For using function relu
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset
batch_size = 64

# PIL Image  28 * 28 * 1, pixel ∈ {0, 255}  咱们读进来的图像一般是 W * H * C
# PyTorch Tensor  1 * 28 * 28, pixel ∈ [0, 1]  转换成 C * W * H 方便进行处理
transform = transforms.Compose([  # Compose 可以将列表里的模块 类似流水线一样对图像进行处理
    transforms.ToTensor(),  # convert the PIL Image to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize(均值, 标准差) 归一化   这里是用经验值 也可以自己算一下均值和标准差
])

train_dataset = datasets.MNIST(root='../../data/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../../data/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 如果不确定reshape成几行 可以指定列, 行只要写-1 会自动算
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不做激活


model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 使用带冲量的优化算法


# training cycle forward, backward, update

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data
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
    with torch.no_grad():  # 说明不需要计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 行是第0个维度 列是第1个维度 找每一行最大值的下标 返回(max, idx)
            total += labels.size(0)  # 取出行数
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
