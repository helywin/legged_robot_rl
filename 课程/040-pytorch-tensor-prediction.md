---
title: PyTorch张量怎样保存并计算数字
aliases:
  - PyTorch tensor 入门
  - 张量预测
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[039-python-class-stores-parameters]]"
  - "[[概念/PyTorch张量]]"
  - "[[概念/Python类与对象]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[学习主页]]"
---

# PyTorch 张量怎样保存并计算数字

## 本课目标

长期方向仍是用 DQN 学习 Chromium B.S.U.，之后再沿另一条路线进入 Go2。本课只把普通 Python 数字换成 PyTorch 张量，完成同一个预测；不学习批次、`nn.Module`、反向传播或参数更新。

## 张量先理解成数字容器

普通 Python 浮点数：

```python
observation = -0.02
```

PyTorch 张量：

```python
observation = torch.tensor(-0.02, dtype=torch.float32)
```

两者都表示 `-0.02`。张量额外记录数据类型和维度，PyTorch 后续可以统一处理许多数字并追踪网络计算。

本课只用三个操作：

1. `torch.tensor(...)`：把数字放进张量；
2. `*`：张量逐项相乘；
3. `.sum()`：把多项贡献相加。

## 同一个预测没有改变含义

上一课的普通数字计算：

```text
观察 × 权重 + 偏置
```

换成张量后仍是：

```python
weighted_observation = observation * weight
prediction = weighted_observation + bias
```

运行：

```bash
.venv/bin/python -m examples.pytorch_tensor_prediction
```

实际输出：

```text
observation=-0.02
weight=+3.00
bias=+0.10
prediction=+0.04
prediction_type=Tensor
prediction_dtype=torch.float32
prediction_dimensions=0
```

`prediction_dimensions=0` 表示结果是只保存一个数的标量张量。它仍是张量，不是 Python `float`。

## 本课训练

打开并只完成一个 `TODO`：

- `exercises/pytorch_tensor_q_prediction.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_tensor_q_prediction
```

示例使用一个观察张量；练习要求把两项观察和两项权重逐项相乘、求和并加上偏置。程序还会检查返回值仍是张量，且输入没有被修改。

## 学习者练习结果

学习者使用 `observation.dot(weights)` 完成两项观察与权重的点积，再加上张量 `bias`。`dot` 等价于逐项相乘后求和，并保持结果为张量。

实际运行结果：

```text
场景1 prediction=+0.500 scalar_tensor=True inputs_unchanged=True PASS
场景2 prediction=+1.600 scalar_tensor=True inputs_unchanged=True PASS
场景3 prediction=+0.500 scalar_tensor=True inputs_unchanged=True PASS
练习通过：你已经用 PyTorch 张量完成两项观察的预测
```

三组结果、返回类型和输入不变性全部通过，因此本课达到完成条件。

## 当前证据边界

> [!success] 启动检查
> PyTorch 2.13.0 CPU 版已经安装；教师示例、学习者练习和自动测试确认张量能完成与普通数字相同的预测，并保持输入不变。

> [!warning] 尚未验证
> 尚未创建 `nn.Module`、多层网络，也没有训练或更新参数。

## 一句话总结

张量没有改变预测含义，它只是把数字放进 PyTorch 能统一计算和管理的容器。

## 关联

- 前置：[[039-python-class-stores-parameters|Python 类保存参数]]
- 概念：[[概念/PyTorch张量]]
- Python 对象：[[概念/Python类与对象]]
- 后续关系：[[概念/神经网络参数与预测]]
- 学习入口：[[学习主页]]
