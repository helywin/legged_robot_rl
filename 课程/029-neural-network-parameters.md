---
title: 神经网络参数是共享的可学习数字
aliases:
  - 神经网络参数
  - 共享参数
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/dqn
  - reinforcement-learning/python
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[024-dqn-replaces-q-table]]"
  - "[[030-predict-action-q-values]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[学习主页]]"
---

# 神经网络参数是共享的可学习数字

## 本课目标

长期方向仍是 DQN 游戏和 Go2 强化学习。本课只理解一个概念：**参数是训练中会改变、并被多个观察共同使用的内部数字。**

## 从 Q 表开始

FrozenLake 的 Q 表为每个“观察—动作”组合单独保存一个值：

```python
q_table[observation][action_index] = q_value
```

16 个观察、4 个动作需要 64 个格子。游戏画面或机器人传感器可能形成数量巨大、甚至连续变化的观察，无法给每种观察单独准备一行。

参数估计改为：

```text
多个观察 → 使用同一组参数计算 → 各自的 Q 值
```

例如 `examples/shared_parameter_q_prediction.py` 为 `LEFT` 和 `RIGHT` 分别保存位置权重、危险程度权重和偏置。这些数字不是某个观察专属的 Q 值，而是一套共享计算规则。

```python
DEFAULT_PARAMETERS = {
    "LEFT": (-0.4, 0.8, 0.1),
    "RIGHT": (0.7, -0.5, 0.0),
}
```

## 职责边界

- 观察是本次输入，不是参数；
- 奖励和目标 Q 值是学习反馈，不是参数；
- 参数保存在估计器内部，训练会逐步修改它们；
- 修改一个共享参数，可能同时改变多个观察的预测。

> [!success] 本课结论
> Q 表把经验分散在独立格子中；参数估计器让多个观察共享一套可调整的数字，因此能处理 Q 表放不下的观察，但一次更新也可能影响多个观察。

## 证据边界

本课使用的仍是纯 Python 线性计算，不是真正的多层神经网络，也没有训练 DQN、Chromium B.S.U.、Isaac Lab 或真机。

## 关联

- 前置：[[024-dqn-replaces-q-table|DQN 怎样替代 Q 表]]
- 下一课：[[030-predict-action-q-values|根据观察预测每个动作的 Q 值]]
- 概念：[[概念/神经网络参数与预测]]
- 上级：[[学习主页]]
