---
title: DQN训练的三种计数
aliases:
  - environment step update count episode count
  - DQN three clocks
tags:
  - knowledge-graph/core
  - reinforcement-learning/dqn
  - reinforcement-learning/training
status: learned
created: 2026-09-04
updated: 2026-09-04
related:
  - "[[概念/强化学习闭环]]"
  - "[[概念/回放预填充与训练起点]]"
  - "[[概念/目标网络]]"
  - "[[概念/DQN完整训练流程与公式]]"
  - "[[065-three-clocks-in-dqn-training]]"
  - "[[066-first-real-cartpole-dqn-smoke-training]]"
---

# DQN 训练的三种计数

DQN 完整训练同时维护三条时间线：

| 计数 | 何时增加 | 回答的问题 |
| --- | --- | --- |
| `environment_step` | 每调用一次环境 `step()` | 收集了多少步新数据 |
| `update_count` | 每执行一次 `optimizer.step()` | 在线参数实际改变了多少次 |
| `episode_count` | 每次环境结束并准备重新 `reset()` | 完成了多少次完整尝试 |

## 关键边界

- 预填充期间 environment step 增加，update count 保持 0；
- 一个 episode 通常包含多个 environment step；
- 一个 environment step 后可以做零次、一次或多次 update；
- 目标同步依据 update count，因为它关心在线参数改变次数；
- episode 结束会重置环境状态，不会清空已经学到的网络参数和全部训练计数。

## 对应课程

- [[065-three-clocks-in-dqn-training|DQN 训练为什么有三种计数]]
- [[066-first-real-cartpole-dqn-smoke-training|第一次真实 CartPole DQN 冒烟训练]]

> [!info]
> 固定六步时间线已经理解；教师参考冒烟训练进一步记录了 `environment_step=30000`、`update_count=29001`、`episode_count=366`。学习者运行结果尚待记录。
