---
title: PyTorch模块与参数
aliases:
  - nn.Module
  - nn.Parameter
  - forward
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
status: learned
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[概念/Python类与对象]]"
  - "[[概念/PyTorch张量]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[概念/自动求导与梯度]]"
  - "[[041-pytorch-module-forward]]"
---

# PyTorch 模块与参数

`nn.Module` 是 PyTorch 模型的基础类。`nn.Parameter` 是被模块登记为模型参数的张量。`forward()` 描述输入张量怎样使用这些参数得到输出张量。

```text
创建 Module 对象
→ __init__ 登记 Parameter
→ model(observation)
→ PyTorch 调用 forward
→ 返回预测张量
```

## 职责边界

- 普通张量可以参与计算，但只有登记后的参数会出现在 `model.parameters()` 中；
- `forward()` 只描述前向计算，不负责修改参数；
- 调用 `model(observation)` 不等于训练；
- 损失、自动求导和优化器属于后续概念。

## 对应课程与代码

- [[041-pytorch-module-forward|nn.Module 怎样组织参数和前向计算]]
- `examples/pytorch_module_prediction.py`
- `exercises/pytorch_module_q_prediction.py`

> [!success] 学习者证据
> 学习者完成模块参数登记和 `forward()`，修正逐项乘法导致的向量输出后，三组标量预测及参数不变性检查全部通过。
