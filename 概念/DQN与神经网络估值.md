---
title: DQN与神经网络估值
aliases:
  - DQN
  - Deep Q Network
tags:
  - knowledge-graph/core
  - reinforcement-learning/dqn
status: future
related:
  - "[[概念/Q表与Q值]]"
  - "[[概念/游戏强化学习路线]]"
---

# DQN 与神经网络估值

当观察数量巨大或连续时，无法为每个观察保存一行 Q 表。DQN 用神经网络接收观察，并输出每个离散动作的 Q 值估计。

它替换的是 Q 值的保存和估计方式，不替换环境、动作、奖励或整个强化学习闭环。

## 对应课程

- [[课程/023-q-learning-for-games|Q-learning 能不能玩游戏]]
- [[课程/024-dqn-replaces-q-table|DQN 怎样替代 Q 表]]

> [!note]
> 当前只建立了概念联系，还没有训练神经网络或 DQN。
