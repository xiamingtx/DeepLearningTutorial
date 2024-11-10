#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:noise_schedule.py
# author:xm
# datetime:2024/7/12 13:35
# software: PyCharm

"""
Noise Schedule 决定了两个分布之间的数据是如何进行转换的，等同于 SGM 的加噪方法。
"""

# import module your need
import torch
from torch import Tensor
from torch.nn import Module
from typing import Dict, Union, Tuple
from utils import align_shape


class FlowNoiser(Module):
    def __init__(
            self, device,
            training_timesteps: int, inference_timesteps: int,
            gamma_min: float, gamma_max: float,
            simplify: bool = True, reparam: str = None
    ):
        """ 采取流式的 noise schedule, 实质就是在两个目标分布之间线性插值. """

        super().__init__()

        self.device = device

        # noise scale 的取值范围
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        # 训练阶段的时间步
        self.training_timesteps = training_timesteps
        self.training_timestep_map = torch.arange(
            0, training_timesteps, 1,
            dtype=torch.long, device=device
        )

        # 推理阶段的时间步
        self.inference_timesteps = inference_timesteps

        # 当推理时的时间步数与训练时不一致时, 可以根据步长映射至训练时的时间步
        self.inference_timestep_map = torch.arange(
            0, training_timesteps, training_timesteps // inference_timesteps,
            dtype=torch.long, device=device
        )

        self.num_timesteps = training_timesteps
        self.timestep_map = self.training_timestep_map

        # 是否是 S-DSB(否则就是 DSB)
        self.simplify = simplify
        if simplify and (reparam is not None) and (reparam not in ("FR", "TR")):
            raise ValueError(f"reparam must be 'FR' or 'TR', got: {reparam}")

        # 所使用的重参数化方式: flow or terminal
        # 在 S-DSB 的情况下才生效
        self.reparam = reparam

    def train(self, mode=True):
        self.num_timesteps = self.training_timesteps if mode else self.inference_timesteps
        self.timestep_map = self.training_timestep_map if mode else self.inference_timestep_map

    def eval(self):
        self.train(mode=False)

    def coefficient(self, t: Union[Tensor, int]) -> Dict:
        """ 用于在两个目标分布之间进行插值的系数. """

        if isinstance(t, Tensor):
            t = t.max()

        if t >= len(self.timestep_map):
            coeff_0, coeff_1 = 0., 1.
        else:
            timestep = self.timestep_map[t].float()
            coeff_1 = timestep / self.training_timesteps
            coeff_0 = 1. - coeff_1

        return {"coeff_0": coeff_0, "coeff_1": coeff_1}

    def prepare_gammas(self):
        """ 使用线性对称的 noise scale,
        也就是先线性递增至最大值; 后线性递减至最小值. """

        gammas = torch.linspace(
            self.gamma_min, self.gamma_max,
            self.num_timesteps // 2,
            device=self.device
        )

        self.gammas = torch.cat([gammas, gammas.flip((0,))])

    def forward(self, x_0: Tensor, x_1: Tensor, t: Union[Tensor, int]) -> Tensor:
        """ 在两个目标分布之间进行插值, 从而得到中间各时间步的状态. """

        coeff = align_shape(x_0, self.coefficient(t))
        coeff_0, coeff_1 = coeff.values()
        x_t = coeff_0 * x_0 + coeff_1 * x_1

        return x_t

    @torch.no_grad()
    def trajectory(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        """ 生成由 x_0 至 x_1 的轨迹. """

        trajectory = [x_0.clone()]
        for t in range(self.num_timesteps):
            x_t = self.forward(x_0, x_1, t)
            trajectory.append(x_t.clone())

        return torch.stack(trajectory)

    def forward_F(self, x: Tensor, x_1: Tensor, t: Union[Tensor, int]) -> Tensor:
        """ DSB 的 F_k^n 函数, mean(x_{k+1}) = F_k^n(x_k)
        在 S-DSB 中被建模为 q 的后验分布, 见论文公式(19)第二式. """
        # x_t = (1 - t) * x_0 + t * x_N, where we denote (1 - t) as coeff_0 and t as coeff_1, corresponding to
        # 1 - \bar\gamma_k and \ba\gamma_k in paper
        coeff_0_t = align_shape(x, self.coefficient(t))["coeff_0"]
        coeff_0_t_plus_one = align_shape(x, self.coefficient(t + 1))["coeff_0"]

        vec = (x_1 - x) / coeff_0_t
        F_x = x + (coeff_0_t - coeff_0_t_plus_one) * vec

        return F_x

    @torch.no_grad()
    def trajectory_F(
            self, x_0: Tensor, x_1: Tensor,
            sample: bool = False
    ) -> Tuple:
        """
        根据 DSB 建模的 forward process(F_k^n 函数) 计算出
        x_0 -> x_1 的整条轨迹. 这也同时可作为 backward model 在
        首个 epoch 里的训练 target.
        returns: Tuple(输入状态，训练目标, 时间步)
         """

        self.prepare_gammas()

        ones = torch.ones((x_0.size(0),), dtype=torch.long, device=self.device)

        x = x_0
        x_all, gt_all, t_all = [], [], []

        for t in range(0, self.num_timesteps):
            ''' 收集轨迹的各时间步 '''
            t_ts = ones * t
            t_all.append(t_ts)

            ''' 收集轨迹的各状态 '''
            # mean(x_{k+1}) = F_k^n(x_k)
            F_x = self.forward_F(x, x_1, t_ts)
            # 如果是采样过程的最后一步, 则不加上噪声项
            # 理论依据是 Tweedie's formula
            if sample and t == self.num_timesteps - 1:
                x_next = F_x
            else:
                # 依据 DSB 的 forward process
                # $ x_{k+1} = F_k^n(x_k) + \sqrt{2 \gamma_{k+1}} \epsilon $
                x_next = F_x + \
                         (2. * self.gammas[t]).sqrt() * torch.randn_like(x)
            x_all.append(x_next.clone())

            ''' 为 backward model 计算并收集轨迹各状态所对应的训练 target '''

            # S-DSB
            if self.simplify:
                if self.reparam == "TR":
                    # terminal 重参数化的预测目标
                    # 参考论文公式(20)第一式
                    gt_all.append(x_0)
                elif self.reparam == "FR":
                    # Flow 重参数化的预测目标
                    # 参考论文公式(21)第一式
                    vec = (x_0 - x_next) / self.coefficient(t + 1)["coeff_1"]
                    gt_all.append(vec)
                else:
                    # 当不使用重参数化时, S-DSB 的预测目标就是前一个状态
                    gt_all.append(x.clone())
            # DSB
            else:
                F_x_next = self.forward_F(x, x_1, t_ts)
                gt_all.append(x_next + F_x - F_x_next)

            x = x_next

        x_all = torch.stack([x_0] + x_all).cpu() if sample else torch.cat(x_all).cpu()
        gt_all = torch.cat(gt_all).cpu()
        t_all = torch.cat(t_all).cpu()

        return x_all, gt_all, t_all


if __name__ == '__main__':
    training_timesteps = 16
    inference_timesteps = 16
    gamma_min = 1e-4
    gamma_max = 1e-3
    simplify = True
    reparam = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    noiser = FlowNoiser(
        device,
        training_timesteps, inference_timesteps,
        gamma_min, gamma_max,
        simplify=simplify, reparam=reparam
    )
