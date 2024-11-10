#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:dsb.py
# author:xm
# datetime:2024/7/12 14:32
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module
from utils import align_shape
from net import ResMLP


class DSB(Module):
    def __init__(
            self, device,
            direction: str, noiser: Module,
            dim_in: int, dim_out: int,
            dim_hidden: int = 128, num_layers: int = 5
    ):
        """ DSB 建模. """

        super().__init__()

        self.device = device

        self.noiser = noiser
        self.num_timesteps = self.noiser.training_timesteps
        self.simplify = self.noiser.simplify
        self.reparam = self.noiser.reparam

        self.noiser.prepare_gammas()
        self.gammas = self.noiser.gammas

        # 指示是 backward or forward model
        self.direction = direction
        if direction not in ('b', 'f'):
            raise ValueError(f"'direction' must be 'b' or 'f', got: {direction}")

        # DSB 所使用的神经网络
        self.network = ResMLP(
            dim_in, dim_out,
            dim_hidden, num_layers,
            n_cond=self.num_timesteps
        ).to(device)

    def forward(self, x_t: Tensor, t: int) -> Tensor:
        """ backward/forward model 在对应时间步所进行预测. """

        timestep = self.noiser.timestep_map[t]
        x = self.network(x_t, timestep)

        return x

    def pred_next(self, x: Tensor, t: Union[Tensor, int]) -> Tensor:
        """ 根据当前状态和时间步预测下一个状态(推理过程).
        对于 backward model 是: x_{k+1} -> x_k, 从 x_1 开始;
        对于 forward model 则是: x_k -> x_{k+1}, 从 x_0 开始. """

        if self.direction == "b":
            dt = -1
            # backward model, 传进来的 t 是从 num_timesteps(=N) - 1 开始,
            # 而状态是从 x_1 开始, 其对应的 \gamma 系数应该是 \gamma_N
            # 因此要 +1
            coeff_t = t + 1
        else:
            dt = 1
            coeff_t = t

        # 当前状态和下一状态的 \bar{\gamma} & 1 - \bar{\gamma} 系数
        coeff_0, coeff_1 = align_shape(x, self.noiser.coefficient(coeff_t)).values()
        coeff_0_next, coeff_1_next = align_shape(x, self.noiser.coefficient(coeff_t + dt)).values()

        # 对于 backward model, 是 x_{k}；
        # 对于 forward model, 是 x_{k+1}.
        pred = self.forward(x, t)
        if self.reparam == "TR":
            if self.direction == "b":
                # bakcward model 预测的终点是 x_0
                x_0 = pred
                # 根据线性插值的原理由预测的终点和当前状态计算出起点
                x_1 = (x - coeff_0 * x_0) / coeff_1
            else:
                # forward model 预测的终点是 x_1
                x_1 = pred
                # 根据线性插值的原理由预测的终点和当前状态计算出起点
                x_0 = (x - coeff_1 * x_1) / coeff_0

            # 根据线性插值的原理由两个端点计算出下一状态
            x_next = coeff_0_next * x_0 + coeff_1_next * x_1
        elif self.reparam == "FR":
            # flow 重参数化情况下, 模型预测的是由当前指向终点的向量.
            vec = pred

            if self.direction == "b":
                # 根據论文公式(19)第一式计算 backward 下一状态的均值
                x_next = x + (coeff_0_next - coeff_0) * vec
            else:
                # 根據论文公式(19)第二式计算 forward 下一状态的均值
                x_next = x + (coeff_1_next - coeff_1) * vec
        else:
            x_next = pred

        return x_next

    @torch.no_grad()
    def inference(self, x: Tensor, sample: bool = False):
        """ 记录推理过程的完整轨迹, 同时也可作为模型训练的 target.

        若是 backward model 的推理, 则记录由 x_1 至 x_0 的各个状态, 同时
        为 forward model 的训练计算 target;

        同理, 若是 forward model 的推理, 则记录由 x_0 至 x_1 各个状态,
        同时为 backward model 的训练计算 target."""

        ones = torch.ones((x.size(0),), dtype=torch.long, device=self.device)

        # 当前 model 推理的起点, 也是另一个 model 的终点
        x_raw = x.clone()
        x_all, gt_all, t_all = [], [], []

        for t in range(self.num_timesteps):
            ''' 收集各时间步 '''

            t_ts = ones * t
            # 若是 backward model 的推理过程,
            # 则方向为 x_1 -> x_0, 时间步由 N-1 至 0
            if self.direction == 'backward':
                t_ts = self.num_timesteps - 1 - t_ts
            t_all.append(t_ts)

            ''' 收集各个状态 '''

            # 若是 backward model, 则是 x_{k+1} -> x_k;
            # 若是 forward model, 则是 x_k -> x_{k+1}.
            mean_x_next = self.pred_next(x, t_ts)

            # 若仅仅是采样过程且到了最后一步, 则不加噪声项
            if sample and t == self.num_timesteps - 1:
                x_next = mean_x_next
            else:
                # 由于 noise scale 呈对称形式, 因此尽管是 backward model,
                # 本来 gamma 的索引应该是 num_timesteps - 1 - t, 这里直接用 t 代替也无影响
                x_next = mean_x_next + \
                         (2. * self.gammas[t]).sqrt() * torch.randn_like(x)
            x_all.append(x_next.clone())

            ''' 为另一个 model 计算并收集 target. '''

            # S-DSB
            if self.simplify:
                if self.reparam == "TR":
                    # 论文公式(20)
                    gt_all.append(x_raw)
                elif self.reparam == "FR":
                    # 论文公式(21)
                    gt_all.append((x_raw - x_next) / ((t + 1) / self.num_timesteps))
                else:
                    # 论文公式(14)
                    gt_all.append(x.clone())
            # DSB
            else:
                mean_mean_x_next = self.pred_next(x_next, t_ts)
                # 论文公式(10)
                gt_all.append(x_next + mean_x_next - mean_mean_x_next)

            x = x_next.clone()

        x_all = torch.stack([x_raw] + x_all).cpu() if sample else torch.cat(x_all).cpu()
        gt_all = torch.cat(gt_all).cpu()
        t_all = torch.cat(t_all).cpu()

        return x_all, gt_all, t_all
