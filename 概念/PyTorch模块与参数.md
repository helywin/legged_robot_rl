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
updated: 2026-09-12
related:
  - "[[概念/Python类与对象]]"
  - "[[概念/PyTorch张量]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[概念/自动求导与梯度]]"
  - "[[041-pytorch-module-forward]]"
  - "[[060-cartpole-four-input-linear-q-network]]"
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

## 能计算、能求导、被优化器管理，要分别确认

| 问题 | 查看什么 |
| --- | --- |
| 数值是否参与本次预测？ | 本次 `forward()` 的实际计算路径 |
| 能否对它求梯度？ | 是否需要梯度、是否处于记录模式、是否连接到 loss |
| 优化器会不会更新它？ | 是否交给该优化器，以及本次梯度和优化器规则 |

普通张量即使设置了 `requires_grad=True`，也不会因此自动成为 `model.parameters()` 的成员。将 `nn.Parameter` 作为模块属性登记后，才能被这种参数遍历找到。模块还可能保存不通过优化器学习的状态，例如归一化统计；所以“网络状态”不完全等于“可学习权重”。

本页简单模型的前向只计算预测；带 BatchNorm 等状态的模型在训练模式前向时还可能更新统计。`model.eval()` 与停止梯度记录也不同，见 [[概念/训练与评估]]。

## 职责边界

- 普通张量可以参与计算，但只有登记后的参数会出现在 `model.parameters()` 中；
- 本页的 `forward()` 描述预测，不在其中执行参数优化；
- 调用 `model(observation)` 不等于训练；
- 第 041 课曾为 VS Code/Pylance 的输出类型提示处理教学模型的 `__call__`，并委托给 `super().__call__()`；这属于当时的类型提示处理，不是每个 PyTorch 模型都必须重写 `__call__`；
- 未完成脚手架尚未给 `self.layer` 赋值时，Pylance 可能依据 `nn.Module` 的动态属性规则推断为 `Tensor | Module`；可以先写 `layer: nn.Linear` 明确属性类型，但这只是静态承诺，运行时仍必须在 `__init__()` 中执行 `self.layer = nn.Linear(...)`；
- 损失、自动求导和优化器属于后续概念。

## 对应课程与代码

- [[041-pytorch-module-forward|nn.Module 怎样组织参数和前向计算]]
- `examples/pytorch_module_prediction.py`
- `exercises/pytorch_module_q_prediction.py`

> [!success] 学习者证据
> 学习者完成模块参数登记和 `forward()`，修正逐项乘法导致的向量输出后，三组标量预测及参数不变性检查全部通过。
