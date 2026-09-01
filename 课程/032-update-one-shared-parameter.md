---
title: 根据误差更新一个共享参数
aliases:
  - 单参数更新
  - 参数向目标移动
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/dqn
  - reinforcement-learning/python
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[031-q-prediction-error]]"
  - "[[033-experience-replay]]"
  - "[[概念/预测误差与参数更新]]"
  - "[[概念/DQN与神经网络估值]]"
  - "[[学习主页]]"
---

# 根据误差更新一个共享参数

## 本课目标

本课只验证：**根据误差调整一个参数，能否让下一次 Q 值预测更接近目标。**

## 最小受控演示

- 问题：一个参数能否根据误差逐步改进预测？
- 固定条件：观察 `0.5`、目标 Q 值 `1.0`、学习率 `0.2`；
- 唯一变化量：参数 `weight`；
- 预测方式：`predicted_q = observation * weight`；
- 成功条件：重复更新时绝对误差持续下降。

更新代码：

```python
predicted_q = observation * weight
error = target_q - predicted_q
weight = weight + learning_rate * error * observation
```

在这个单参数例子中，`observation` 表示参数对当前预测的影响：观察为正时，增大参数也会增大预测。学习率控制每次只走一小步。

## 实际运行

```bash
.venv/bin/python -m examples.single_parameter_q_update
```

```text
更新前0: weight=1.000000 prediction=0.500000 error=+0.500000
更新前1: weight=1.050000 prediction=0.525000 error=+0.475000
更新前2: weight=1.097500 prediction=0.548750 error=+0.451250
更新前3: weight=1.142625 prediction=0.571313 error=+0.428687
更新前4: weight=1.185494 prediction=0.592747 error=+0.407253
更新前5: weight=1.226219 prediction=0.613110 error=+0.386890
```

预测从 `0.500` 向目标 `1.000` 移动，绝对误差从 `0.500` 连续下降到 `0.387`，达到本演示的成功条件。

## 为什么只移动一小步

共享参数可能同时影响多个观察。一次经验只提供局部信息，如果直接把当前预测改成目标值，可能严重破坏其他观察上的估计。学习率让参数在多条经验之间逐步调整。

真实神经网络有许多参数，程序还要计算每个参数分别怎样影响预测。本课只建立单参数更新直觉，不提前展开完整反向传播。

> [!success] 当前证据
> 两个标准库示例均能启动，新增 9 项单元测试通过。这里只验证共享参数预测和单参数误差下降，不是神经网络训练、DQN 冒烟训练、游戏评测、Isaac Lab 仿真或真机验证。

## 关联

- 前置：[[031-q-prediction-error|所选动作的 Q 值预测误差]]
- 概念：[[概念/预测误差与参数更新]]
- 后续上级概念：[[概念/DQN与神经网络估值]]
- 下一课：[[033-experience-replay|经验回放：保存后随机重用旧经验]]
- 上级：[[学习主页]]
