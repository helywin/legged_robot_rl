---
title: 所选动作Q值与索引
aliases:
  - Selected action Q value
  - 动作索引取值
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: learned
created: 2026-09-01
updated: 2026-09-02
related:
  - "[[概念/线性层与多动作Q值]]"
  - "[[概念/预测误差与参数更新]]"
  - "[[概念/DQN训练流程]]"
  - "[[047-pytorch-selected-action-q-loss]]"
  - "[[概念/DQN目标Q值]]"
  - "[[概念/批量张量与逐行动作索引]]"
---

# 所选动作 Q 值与索引

离散动作 Q 网络会输出每个动作的 Q 值，但一条经验只直接评价实际执行的动作。训练时使用动作编号从输出张量中取出对应标量：

```python
q_values = model(observation)
selected_q = q_values[executed_action]
```

如果探索阶段执行的动作不是当前最大 Q 值动作，也必须使用实际动作编号，不能用 `argmax()` 偷换成当前最佳动作。

## 梯度关系

损失只由 `selected_q` 计算时，计算图只把这次直接误差连到对应输出。对于一个没有隐藏层的两输出线性层，未选择动作对应的输出行梯度为零，所选动作行得到非零梯度。

## 职责边界

- `executed_action` 来自实际环境交互记录，不是重新选择出来的动作；
- 索引后的 `selected_q` 必须保持为张量，转换成 Python 数值会断开自动求导；
- 本节点只选择预测值，目标 Q 值怎样由奖励和下一观察形成属于后续步骤；
- 输出最大值用于利用决策，实际动作索引用于训练当前经验，二者职责不同。

## 对应课程与代码

- [[047-pytorch-selected-action-q-loss|只取实际动作的 Q 值计算损失]]
- `examples/pytorch_selected_action_q_loss.py`
- `exercises/pytorch_selected_action_q_loss.py`

学习者已经完成实际练习：当前最大 Q 值对应动作 0，但经验实际执行动作 1；损失正确使用动作 1 的 `-0.05`，反向后未选择行梯度为零、所选行梯度非零。

单条经验可以直接使用一个动作编号；多条经验组成二维 Q 值矩阵后，必须为每行选择各自的动作列，见 [[概念/批量张量与逐行动作索引]]。
