"""固定CPU SGD更新的编译执行路径；学习版Agent保留为可对照的实现。

仍使用PyTorch自动求导：前向→目标→MSE→autograd.grad→SGD。
autograd.grad直接返回梯度，不写参数.grad；编译循环执行p -= lr * grad，
等价于本项目无momentum/weight_decay的SGD。其他优化器配置明确拒绝。
不改变经验抽样、更新次数、目标网络同步或推理权重格式。
本机Python3.14/PyTorch2.13实测一致，但torch.jit.script发出兼容性警告；
因此仅作显式选择的实验后端，普通eager保持默认。
"""
import torch
from torch import nn, Tensor
from agent import Agent, Batch, UpdateStats


class _SGDUpdate(nn.Module):
    def __init__(self, online: nn.Module, target: nn.Module) -> None:
        super().__init__()
        self.online = online
        self.target = target
        self.params = list(online.parameters())

    def forward(self, observations: Tensor, actions: Tensor, rewards: Tensor,
                next_observations: Tensor, terminated: Tensor,
                gamma: float, learning_rate: float) -> tuple[Tensor, Tensor, Tensor]:
        selected = self.online(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            future = self.target(next_observations).max(1).values
            expected = rewards + future * (~terminated).float() * gamma
        loss = torch.nn.functional.mse_loss(expected, selected)
        gradients = torch.autograd.grad([loss], self.params)
        with torch.no_grad():
            for parameter, gradient in zip(self.params, gradients):
                if gradient is not None:
                    parameter.add_(gradient, alpha=-learning_rate)
        return loss.detach(), selected.detach().mean(), expected.mean()


class ScriptedSGDAgent(Agent):
    def __init__(self, online: nn.Module, target: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float = 0.99,
                 exploration_seed: int = 11) -> None:
        super().__init__(online, target, optimizer, gamma, exploration_seed)
        if type(optimizer) is not torch.optim.SGD or len(optimizer.param_groups) != 1:
            raise ValueError('编译更新仅支持一个参数组的普通SGD')
        group = optimizer.param_groups[0]
        if any(group.get(key, False) for key in
               ('momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'differentiable', 'foreach', 'fused')):
            raise ValueError('编译更新不支持动量、权重衰减或其他SGD扩展')
        parameters: list[nn.Parameter] = list(online.parameters())
        if [id(p) for p in group['params']] != [id(p) for p in parameters]:
            raise ValueError('优化器必须按网络顺序管理全部在线参数')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 or not p.requires_grad
               for p in parameters):
            raise ValueError('编译更新仅支持CPU float32可训练参数')
        self._kernel = torch.jit.script(_SGDUpdate(online, target))

    def update(self, batch: Batch) -> UpdateStats:
        loss, prediction, expected = self._kernel(
            batch.observations, batch.actions, batch.rewards,
            batch.next_observations, batch.terminated,
            self.gamma, float(self.optimizer.param_groups[0]['lr']),
        )
        self.updates += 1
        return UpdateStats(loss.item(), prediction.item(), expected.item(), self.updates)
