#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:vae.py
# author:xm
# datetime:2024/3/20 14:14
# software: PyCharm

"""
https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
"""

# import module your need
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.enc_mu = torch.nn.Linear(H, latent_size)
        self.enc_log_sigma = torch.nn.Linear(H, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.enc_mu(x)
        # sigma > 0, if no regularization, it's difficult to achieve positive sigma,
        # so output logvar and apply exp to reconstruct it
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        # gaussian distribution with mean=mu and var=sigma
        return torch.distributions.Normal(loc=mu, scale=sigma)


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        # deconstruct image from noise sampled in gaussian distribution
        x = F.relu(self.linear1(x))
        mu = torch.tanh(self.linear2(x))
        # gaussian distribution with mean=mu and var=1
        return torch.distributions.Normal(loc=mu, scale=torch.ones_like(mu))


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()
        return self.decoder(z), q_z


transform = transforms.Compose(
    [transforms.ToTensor(),
     # Normalize the images to be -0.5, 0.5
     transforms.Normalize(0.5, 1)]
)
mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

input_dim = 28 * 28
batch_size = 128
num_epochs = 100
learning_rate = 0.001
hidden_size = 512
latent_size = 8

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dataloader = torch.utils.data.DataLoader(
    mnist, batch_size=batch_size,
    shuffle=True,
    pin_memory=torch.cuda.is_available())

print('Number of samples: ', len(mnist))

encoder = Encoder(input_dim, hidden_size, latent_size)
decoder = Decoder(latent_size, hidden_size, input_dim)

vae = VAE(encoder, decoder).to(device)

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.view(-1, input_dim).to(device)
        optimizer.zero_grad()
        p_x, q_z = vae(inputs)
        # log_prob: 使用对数似然来衡量模型生成的数据与真实数据的匹配程度
        log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
        # 让q_z尽量符合标准正态分布 (prior assumption)
        kl = torch.distributions.kl_divergence(
            q_z,
            torch.distributions.Normal(0, 1.)
        ).sum(-1).mean()
        loss = -(log_likelihood - kl)
        loss.backward()
        optimizer.step()
        l = loss.item()
    print(epoch, l, log_likelihood.item(), kl.item())
