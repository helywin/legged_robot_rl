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
updated: 2026-09-12
related:
  - "[[概念/强化学习闭环]]"
  - "[[概念/回放预填充与训练起点]]"
  - "[[概念/目标网络]]"
  - "[[概念/DQN完整训练流程与公式]]"
  - "[[065-three-clocks-in-dqn-training]]"
  - "[[066-first-real-cartpole-dqn-smoke-training]]"
---

# DQN 训练的三种计数

DQN 完整训练同时维护三条时间线。先按早期单环境实现解释：

| 计数 | 何时增加 | 回答的问题 |
| --- | --- | --- |
| `environment_step` | 每调用一次环境 `step()` | 收集了多少步新数据 |
| `update_count` | 每执行一次约定的优化器更新 | 尝试更新了多少次；不保证每个参数都变化 |
| `episode_count` | 每次环境结束并准备重新 `reset()` | 完成了多少次完整尝试 |

## 关键边界

- 预填充期间 environment step 增加，update count 保持 0；
- 一个 episode 通常包含多个 environment step；
- 一个 environment step 后可以做零次、一次或多次 update；
- 本仓库早期练习按 update count 同步目标网络；其他实现也可能按交互步计时，必须查调度条件；
- episode 结束会重置环境状态，不会清空已经学到的网络参数和全部训练计数。

## 并行环境会让“调用一次”与“数据一条”分开

假设四个环境一起推进，三次并行调用中每次都真实产生四条转移，没有纯重置步，则共采集 4 × 3 = 12 条经验。若随后做两次网络更新，而期间一个子环境结束了一局：

- 并行 `step()` 调用次数：3；
- 新转移条数：12；
- 优化器更新次数：2；
- 结束的回合数：1。

日志里的 `steps` 可能表示其中不同的单位，比较预算前要查清定义。动作保持又会引入物理小步，见 [[概念/动作保持与观察时间对齐]]。不能把“更新 1000 次”直接写成“游戏玩了 1000 步”。

## 对应课程

- [[065-three-clocks-in-dqn-training|DQN 训练为什么有三种计数]]
- [[066-first-real-cartpole-dqn-smoke-training|第一次真实 CartPole DQN 冒烟训练]]

> [!info]
> 第 065—066 课当时记录：固定六步时间线已经理解；教师参考冒烟训练为 `environment_step=30000`、`update_count=29001`、`episode_count=366`。这组教师数据不能冒充学习者的新运行结果，当前状态应查课程原记录。
