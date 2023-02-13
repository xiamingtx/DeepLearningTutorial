#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:BasicCNNExercise.py
# author:xm
# datetime:2023/2/13 13:29
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
import torch.optim as optim  # 优化器优化
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

batch_size = 64
# transform预处理,把图像转化成图像张量
'''ToTensor：将一个’PIL Image‘or‘numpy.ndarray’转化为tensor格式，PIL Image or numpy.ndarray的shape为(H x W x C)，范围是[0, 255]
   转化为shape为(C x H x W)范围在[0.0, 1.0]'''

'''Normalize：标准化函数，使用均值和标准差对tensor进行标准化
   output[channel] = (input[channel] - mean[channel]) / std[channel]
'''
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# mnist数据集为1*28*28的单通道图像
train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               download=True,
                               transform=transform)  # 训练数据集
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 32, kernel_size=3)
        self.pooling = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 10)

    def forward(self, x):
        # (batch, 1, 28, 28) -> (batch, 10, 26, 26) -> (batch, 10, 13, 13)
        x = self.pooling(self.relu(self.conv1(x)))
        # (batch, 10, 13, 13) -> (batch, 20, 13, 13) -> (batch, 20, 6, 6)
        x = self.pooling(self.relu(self.conv2(x)))
        # (batch, 20, 6, 6) -> (batch, 32, 4, 4) -> (batch, 32, 2, 2)
        x = self.pooling(self.relu(self.conv3(x)))
        # 此时x是维度为4的tensor，即(batchsize，C，H，W)，x.size(0)指batchsize的值
        # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
        # (batch, 32, 2, 2) -> (batch, 128)
        x = x.view(x.size(0), -1)
        # (batch, 128) -> (batch, 64) -> (batch, 32) -> (batch, 10)
        x = self.fc1(x)  # x应用全连接网络
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 因为网络模型已经有点大了，所以梯度下降里面要用更好的优化算法，比如用带冲量的（momentum），来优化训练过程


# 把一轮循环封装到函数里面
def train(epoch):
    running_loss = 0.0
    # 通过函数enumerate返回每一批数据data，以及索引index(batch_idx)，因为start=0所以index从0开始
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data  # 将数据分为图像以及所对应的标签
        inputs, targets = inputs.to(device), targets.to(device)  # 迁移到GPU上
        optimizer.zero_grad()  # 将历史损失梯度清零
        # forward
        y_pred = model(inputs)  # 将训练图片输入网络得到输出
        # backward
        loss = criterion(y_pred, targets)
        loss.backward()
        # update
        optimizer.step()  # 参数更新
        running_loss += loss.item()
        if batch_idx % 300 == 299:  # 每300个mini-batches打印一次
            print('[%d,%5d] loss：%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0  # 正确多少
    total = 0  # 总数多少
    with torch.no_grad():  # 测试不用算梯度
        for data in test_loader:  # 从test_loader拿数据
            images, labels = data  # 将数据分为图像以及所对应的标签
            images, labels = images.to(device), labels.to(device)  # 迁移到GPU上
            outputs = model(images)  # 拿完数据做预测
            _, predicted = torch.max(outputs.data, dim=1)  # 沿着第一个维度找最大值的下标，返回值有两个，因为是10列嘛，返回值
            # 返回值一个是每一行的最大值，另一个是最大值的下标（每一个样本就是一行，每一行有10个量）（行是第0个维度，列是第1个维度）
            total += labels.size(0)  # 取size元组的第0个元素（N，1），累加后可得测试样本的总数
            # 推测出来的分类与label是否相等，真就是1，假就是0，求完和之后把标量拿出来，循环完成后可得预测正确的样本数
            correct += (predicted == labels).sum().item()
    print("accuracy on test set:%d %% [%d/%d]" % (100 * correct / total, correct, total))


# 训练
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()  # 训练一轮，测试一轮
