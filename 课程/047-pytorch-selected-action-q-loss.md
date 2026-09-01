---
title: 只取实际动作的Q值计算损失
aliases:
  - PyTorch selected action Q
  - 动作索引损失
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[046-pytorch-two-action-q-values]]"
  - "[[031-q-prediction-error]]"
  - "[[概念/所选动作Q值与索引]]"
  - "[[概念/预测误差与参数更新]]"
  - "[[学习主页]]"
  - "[[048-pytorch-dqn-target-q]]"
---

# 只取实际动作的 Q 值计算损失

## 本课目标

长期方向仍是先完成 CartPole 小型 DQN，再走向 Chromium B.S.U.，并为 Go2 路线保留共同基础。本课只学习从全部动作 Q 值中取出实际执行动作对应的一个值，再对它计算损失；不计算完整 DQN 目标，也不更新参数。

## 为什么不能总取最大 Q 值

模型预测：

```text
q_values = [0.10, -0.05]
```

动作 0 当前分数更高。但假设 ε-greedy 为了探索，实际执行了动作 1。这条经验得到的反馈评价的是动作 1，所以训练时必须选择：

```text
executed_action = 1
selected_q = q_values[1] = -0.05
```

如果改用 `argmax()` 取 `0.10`，就会把动作 1 的反馈错误地记到动作 0 上。

## 使用张量索引

代码只需要：

```python
selected_q = q_values[executed_action]
```

`selected_q` 是零维标量张量，仍然连接计算图。不要在计算损失前调用 `.item()`，否则它会变成普通 Python 数值，自动求导无法沿它回到模型参数。

固定目标为 `0.50` 时：

```text
loss = (-0.05 - 0.50)² = 0.3025
```

## 梯度流向哪里

本课模型没有隐藏层，两个输出各对应线性层的一行参数。损失只使用动作 1 的输出，因此反向计算后：

```text
动作0参数行梯度 = [0.0, 0.0]
动作1参数行梯度 = [-0.22, 0.11]
```

这不是说动作 0 永远不学习，而是这条经验只直接训练实际执行的动作 1。

运行教师示例：

```bash
.venv/bin/python -m examples.pytorch_selected_action_q_loss
```

## 本课训练

打开并完成 `selected_action_loss()` 中的一个 `TODO`：

- `exercises/pytorch_selected_action_q_loss.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_selected_action_q_loss
```

你需要预测全部 Q 值、按 `executed_action` 索引、计算平方损失并调用 `backward()`。检查器会确认没有用 `argmax()` 偷换实际动作，并观察两行参数的梯度。

## 学习者练习结果

学习者按 `executed_action=1` 选择了第二个 Q 值，没有使用当前最佳动作 `0`。实际运行结果：

```text
q_values=[0.1, -0.05] PASS
best_action=0 PASS
executed_action=1 PASS
selected_q=-0.050 PASS
loss=0.3025 PASS
unselected_row_grad_zero=True PASS
selected_row_grad_nonzero=True PASS

练习通过：损失只连接到实际执行动作的 Q 值
```

这确认了学习者能够区分“利用时选择最大 Q 值”和“训练经验时使用实际动作索引”两种职责，本课达到完成条件。

## 当前边界

> [!success] 启动检查
> 教师示例、学习者练习和自动测试确认损失使用实际动作 Q 值，并只把直接梯度传到所选输出行。

> [!warning] 尚未验证
> target 仍是人工给定，没有奖励、下一观察、目标网络、optimizer 更新、CartPole DQN 或独立评估。

## 一句话总结

一条经验训练的是实际执行动作对应的 Q 值，不能用当前最大 Q 值替代实际动作索引。

## 关联

- 前置：[[046-pytorch-two-action-q-values|一组观察怎样输出两个动作 Q 值]]
- 旧知识：[[031-q-prediction-error|所选动作的 Q 值预测误差]]
- 概念：[[概念/所选动作Q值与索引]]
- 参数更新：[[概念/预测误差与参数更新]]
- 学习入口：[[学习主页]]
- 下一课：[[048-pytorch-dqn-target-q|奖励和下一观察怎样形成 DQN 目标值]]
