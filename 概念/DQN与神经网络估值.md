---
title: DQN与神经网络估值
aliases:
  - DQN
  - Deep Q Network
tags:
  - knowledge-graph/core
  - reinforcement-learning/dqn
status: learning
updated: 2026-09-01
related:
  - "[[概念/Q表与Q值]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[概念/预测误差与参数更新]]"
  - "[[概念/经验回放]]"
  - "[[概念/目标网络]]"
  - "[[概念/DQN训练流程]]"
  - "[[概念/Python函数模型]]"
  - "[[概念/游戏强化学习路线]]"
---

# DQN 与神经网络估值

当观察数量巨大或连续时，无法为每个观察保存一行 Q 表。DQN 用神经网络接收观察，并输出每个离散动作的 Q 值估计。

它替换的是 Q 值的保存和估计方式，不替换环境、动作、奖励或整个强化学习闭环。

## 对应课程

- [[课程/023-q-learning-for-games|Q-learning 能不能玩游戏]]
- [[课程/024-dqn-replaces-q-table|DQN 怎样替代 Q 表]]
- [[课程/029-neural-network-parameters|神经网络参数是共享的可学习数字]]
- [[课程/030-predict-action-q-values|根据观察预测每个动作的 Q 值]]
- [[课程/031-q-prediction-error|所选动作的 Q 值预测误差]]
- [[课程/032-update-one-shared-parameter|根据误差更新一个共享参数]]
- [[课程/033-experience-replay|经验回放：保存后随机重用旧经验]]
- [[课程/034-target-network|目标网络：暂时固定训练目标]]
- [[课程/035-one-dqn-training-step|一条回放经验怎样完成一次 DQN 更新]]
- [[课程/038-python-function-as-model|Python 函数最小预测]]

> [!note]
> 当前退回标准库 Python 函数层补齐神经网络使用前置。只有学习者完成对应练习后，才进入 Python 类和 PyTorch 网络。
