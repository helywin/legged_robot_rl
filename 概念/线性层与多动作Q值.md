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
updated: 2026-09-03
related:
  - "[[概念/神经网络参数与预测]]"
  - "[[概念/PyTorch模块与参数]]"
  - "[[概念/DQN与神经网络估值]]"
  - "[[046-pytorch-two-action-q-values]]"
  - "[[060-cartpole-four-input-linear-q-network]]"
  - "[[概念/隐藏层与非线性]]"
  - "[[061-linear-limit-relu-hidden-layer]]"
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

## CartPole 的四输入扩展

CartPole 每条观察有四项，动作仍然只有两个，因此对应线性层是 `nn.Linear(4, 2)`：

```text
weight.shape = (2, 4)
bias.shape = (2,)
参数总数 = 2 × 4 + 2 = 10
```

输入数量和输出数量由环境接口决定，不是训练自动发现的。输入批次形状可以是 `(batch_size, 4)`，线性层会让每一行分别得到两个 Q 值。最终 Q 值允许为负，因此输出层不使用 softmax，也不应为了隐藏负值而强制 ReLU。

若输出形状为 `(batch_size, action_count)`，为每条经验分别选动作应使用 `argmax(dim=1)`，即在每一行的动作列之间比较。`argmax(dim=0)` 比较的是不同经验行，回答的是每个动作在哪条经验上最高，不是每条经验选择哪个动作。

- [[060-cartpole-four-input-linear-q-network|CartPole 四项观察怎样变成两个动作 Q 值]]

连续两个线性层若中间没有激活函数，仍可合并成一个线性层。需要表达随输入区域变化的分段关系时，可在线性隐藏层后加入 ReLU；最终 Q 值输出层仍保持线性。详见 [[概念/隐藏层与非线性]]。
