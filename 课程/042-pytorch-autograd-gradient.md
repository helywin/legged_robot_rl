---
title: 自动求导怎样计算参数梯度
aliases:
  - PyTorch backward 入门
  - 损失与梯度
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
status: learning
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[041-pytorch-module-forward]]"
  - "[[概念/自动求导与梯度]]"
  - "[[概念/PyTorch模块与参数]]"
  - "[[概念/预测误差与参数更新]]"
  - "[[学习主页]]"
---

# 自动求导怎样计算参数梯度

## 本课目标

长期方向仍是用 DQN 学习 Chromium B.S.U.，之后再沿另一条路线进入 Go2。本课只让 PyTorch 根据一次预测损失计算参数梯度；不使用优化器，不修改参数，也不运行 DQN。

## 从预测到损失

当前模型得到预测 `0.5`，目标是 `1.0`。先用普通话说：预测没有达到目标，需要一个数字表示“差得有多严重”。这个数字叫损失。

本课使用平方损失：

```text
先求差：预测 - 目标 = 0.5 - 1.0 = -0.5
再平方：(-0.5)² = 0.25
```

代码：

```python
loss = (prediction - target) ** 2
```

平方让正负误差都变成非负损失，而且偏差越大，损失增长越快。

## gradient 表示什么

梯度可以先理解为：如果某个参数稍微增大，损失会朝哪个方向变化、变化有多敏感。

- 梯度为正：增大参数会让损失增大，训练通常应减小参数；
- 梯度为负：增大参数会让损失减小，训练通常应增大参数；
- 绝对值越大：损失对该参数越敏感。

调用：

```python
loss.backward()
```

PyTorch 会沿着刚才的预测计算反向追踪，并把结果写到参数的 `.grad` 中。

## 可见示例

固定数据：

```text
observation=0.5
weight=1.0
bias=0.0
prediction=0.5
target=1.0
```

运行：

```bash
.venv/bin/python -m examples.pytorch_autograd_demo
```

实际输出：

```text
observation=+0.50
prediction=+0.50
target=+1.00
loss=+0.25
weight_grad=-0.50
bias_grad=-1.00
parameters_changed=False
```

两个梯度都是负数，说明在当前这条数据上，增大权重或偏置会让预测靠近目标、损失下降。但 `backward()` 只计算梯度，没有修改参数，所以 `parameters_changed=False`。

## 本课训练

打开并只完成一个 `TODO`：

- `exercises/pytorch_autograd_q_error.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_autograd_q_error
```

练习要求通过 `model(observation)` 完成预测，再计算平方损失、调用 `backward()` 并返回结果。程序会检查模型的 `forward()` 确实被调用、预测、损失、两个梯度以及参数保持不变。本课保持 `status: learning`，看到你的代码和通过输出后才会完成。

## 当前证据边界

> [!success] 启动检查
> 教师示例和自动测试确认 PyTorch 能从平方损失计算两个参数梯度，且 `backward()` 本身不会更新参数。

> [!warning] 尚未验证
> 学习者练习尚未完成；没有学习率、优化器、参数更新、训练循环或 DQN。

## 一句话总结

损失表示预测与目标的差距，`backward()` 根据计算过程把每个参数的梯度写进 `.grad`，但不会自动修改参数。

## 关联

- 前置：[[041-pytorch-module-forward|nn.Module 前向计算]]
- 概念：[[概念/自动求导与梯度]]
- 模块参数：[[概念/PyTorch模块与参数]]
- 后续更新：[[概念/预测误差与参数更新]]
- 学习入口：[[学习主页]]
