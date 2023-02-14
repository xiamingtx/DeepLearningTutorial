#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:NameClassifier.py
# author:xm
# datetime:2023/2/13 22:28
# software: PyCharm

"""
homework: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
"""

# import module your need
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import gzip
import csv

from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')  # 可以忽略matplotlib的warning


# prepare dataset
class NameDataset(Dataset):

    def __init__(self, is_train_set=True):
        # 从gz当中读取数据
        filename = '../dataset/names_train.csv.gz' if is_train_set else '../dataset/names_test.csv.gz'

        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)  # 每一行都是(name,country)的元组
            rows = list(reader)
        # 将names和countries保存在list中
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        # 将countries和它的index保存在list和dictionary中
        self.country_list = list(sorted(set(self.countries)))  # 每个国家只剩一个实例
        self.country_dict = self.getCountryDict()  # 转换成字典 例如{'Arabic': 0, 'Chinese': 1}
        self.country_num = len(self.country_list)  # 国家的个数

    # 提供索引访问
    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]  # 返回(姓名, 国家索引)

    # 返回dataset的长度
    def __len__(self):
        return self.len

    # 将list转化成dictionary
    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    # 给定index返回country，方便展示
    def idx2country(self, index):
        return self.country_list[index]

    # 返回country的数目
    def getCountriesNum(self):
        return self.country_num


# Prepare Dataset and DataLoader
# Parameters
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
USE_GPU = False
# 训练数据
trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
# 测试数据
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
# N_COUNTRY is the output size of our model
N_COUNTRY = trainset.getCountriesNum()


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


# define model
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        # parameters of GRU layer
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # What is the Bi-Direction RNN/LSTM/GRU?
        self.n_directions = 2 if bidirectional else 1

        # The input of Embedding Layer with shape:𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒
        # The output of Embedding Layer with shape:𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # The inputs of GRU Layer with shape:
        # 𝑖𝑛𝑝𝑢𝑡: 𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒
        # ℎ𝑖𝑑𝑑𝑒𝑛: 𝑛𝐿𝑎𝑦𝑒𝑟𝑠 ∗ 𝑛𝐷𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛𝑠, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒
        # The outputs of GRU Layer with shape:
        # 𝑜𝑢𝑡𝑝𝑢𝑡: 𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒 ∗ 𝑛𝐷𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛𝑠
        # ℎ𝑖𝑑𝑑𝑒𝑛: 𝑛𝐿𝑎𝑦𝑒𝑟𝑠 ∗ 𝑛𝐷𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛𝑠, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        # input shape : B x S -> S x B
        input = input.t()
        # Save batch-size for make initial hidden
        batch_size = input.size(1)

        # Initial hidden with shape:
        # (𝑛𝐿𝑎𝑦𝑒𝑟 ∗ 𝑛𝐷𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛𝑠, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒)
        hidden = self._init_hidden(batch_size)
        # Result of embedding with shape:
        # (𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒)
        embedding = self.embedding(input)

        # pack them up
        # The first parameter with shape:
        # (𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒)
        # The second parameter is a tensor, which is a list of sequence length of each batch element.
        # Result of embedding with shape:(𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒)
        # It returns a PackedSequence object.
        gru_input = pack_padded_sequence(embedding, seq_lengths)  # tips，用GPU时需要在此处更改：在seq_lengths加一个.cpu()
        # The output is a PackedSequence object, actually it is a tuple.
        # the shape of hidden, which we concerned, with shape:
        # (𝑛𝐿𝑎𝑦𝑒𝑟𝑠 ∗ 𝑛𝐷𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒)
        output, hidden = self.gru(gru_input, hidden)

        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        fc_output = self.fc(hidden_cat)
        return fc_output


# name2list是一个元组
def name2list(name):
    # 把名字变为列表，列表生成式↓👇，把每一个名字变成一个ASCII码列表
    arr = [ord(c) for c in name]
    # 返回一个列表本身和列表的长度
    return arr, len(arr)


def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    # 取出列表的名字和列表长度
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()

    # make tensor of name, BatchSize x SeqLen
    # 先做一个全0的张量
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    # 这是一个复制操作，
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    # 排完序后得到 seq_lengths（排序后的序列） perm_idx（排序后对应的ID，即索引）
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
        print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
        print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss


def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total


if __name__ == '__main__':
    # N_CHARS：字符数量（输入的是英文字母，每一个字符都要转变成one-hot向量，这是自己设置的字母表的大小）
    # HIDDEN_SIZE：隐层数量（GRU输出的隐层的维度）
    # N_COUNTRY：一共有多少个分类
    # N_LAYER：设置用几层的GRU
    # 实例化分类模型

    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)  # instantiate the classifier model

    # 判断是否使用GPU训练模型
    if USE_GPU:
        device = torch.device("cuda:0")

        classifier.to(device)

    # 构造损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()  # 计算一下时间
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    # 每一次epoch做一次训练和测试
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        trainModel()
        acc = testModel()
        # 测试结果添加到acc_list列表，可以绘图等
        acc_list.append(acc)

    epoch = np.arange(1, len(acc_list) + 1, 1)

    acc_list = np.array(acc_list)

    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
