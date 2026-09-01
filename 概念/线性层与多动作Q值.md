---
title: 线性层与多动作Q值
aliases:
  - nn.Linear
  - 多动作输出层
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: learned
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[概念/神经网络参数与预测]]"
  - "[[概念/PyTorch模块与参数]]"
  - "[[概念/DQN与神经网络估值]]"
  - "[[046-pytorch-two-action-q-values]]"
  - "[[概念/所选动作Q值与索引]]"
---

# 线性层与多动作 Q 值

`nn.Linear` 是 PyTorch 提供的线性计算层。`nn.Linear(2, 2)` 接收两个观察特征，并输出两个数字；在离散动作 DQN 中，可以把两个输出依次解释为两个动作的 Q 值。

```text
[观察量0, 观察量1]
        ↓ nn.Linear(2, 2)
[动作0的Q值, 动作1的Q值]
```

线性层内部登记了权重矩阵和偏置：

```text
weight.shape = (输出数量, 输入数量) = (2, 2)
bias.shape = (输出数量,) = (2,)
```

## 职责边界

- 输出顺序必须和动作编号约定一致；
- Q 值是未来回报估计，不是概率，可以为负数，因此不使用 softmax；
- `argmax()` 只是利用当前最大 Q 值选择动作，探索仍需后续的 ε-greedy；
- 当前只有一个线性层，还没有隐藏层或完整 DQN 训练。

## 对应课程与代码

- [[046-pytorch-two-action-q-values|一组观察怎样输出两个动作 Q 值]]
- `examples/pytorch_two_action_q_values.py`
- `exercises/pytorch_two_action_q_output.py`

学习者已经完成实际练习：创建两输入两输出的 `nn.Linear`，并用三组观察确认输出顺序、最佳动作变化和预测不修改参数。
