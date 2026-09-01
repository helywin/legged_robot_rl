---
title: PyTorch张量
aliases:
  - Tensor
  - torch.tensor
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
status: learned
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[概念/Python函数模型]]"
  - "[[概念/Python类与对象]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[概念/PyTorch模块与参数]]"
  - "[[040-pytorch-tensor-prediction]]"
---

# PyTorch 张量

张量是 PyTorch 用来保存和计算数字的容器。它除了数值，还记录数据类型和维度。标量张量只保存一个数，长度为 2 的一维张量可以保存两项观察。

张量逐项乘法会让相同位置的数字分别相乘；`.sum()` 再把多项结果相加：

```text
[观察1, 观察2] × [权重1, 权重2]
→ [贡献1, 贡献2]
→ sum
→ 一个标量预测张量
```

## 职责边界

- 张量是数据容器，不等于神经网络；
- 创建张量不等于训练；
- `.item()` 可以读取单元素张量中的 Python 数值，但网络计算通常应继续保留张量；
- 批次、自动求导和参数更新属于后续概念。

## 对应课程与代码

- [[040-pytorch-tensor-prediction|PyTorch 张量怎样保存并计算数字]]
- `examples/pytorch_tensor_prediction.py`
- `exercises/pytorch_tensor_q_prediction.py`

> [!success] 学习者证据
> 学习者使用张量点积和偏置完成三组预测，返回值类型、零维形状与输入不变性检查全部通过。
