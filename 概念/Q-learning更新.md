---
title: Q-learning更新
aliases:
  - Q-learning update
  - 一次 Q 值更新
tags:
  - knowledge-graph/core
  - reinforcement-learning/q-learning
status: active
related:
  - "[[概念/Q表与Q值]]"
  - "[[概念/环境奖励与回合]]"
  - "[[概念/训练与评估]]"
  - "[[概念/Q表完整训练流程与公式]]"
---

# Q-learning 更新

一条经验包含旧观察、动作、奖励、新观察和是否终止。Q-learning 用它计算目标值，再让旧 Q 值按学习率朝目标值移动。

如果任务真正终止，目标只看当前奖励；否则还要参考新观察中最大的 Q 值，并由折扣因子决定未来价值的权重。

## 对应课程

- [[课程/013-update-q-from-one-reward|一次奖励如何更新 Q 值]]
- [[课程/014-future-value-and-discount|未来价值与折扣]]
- [[课程/015-one-q-learning-update|完整的一次 Q-learning 更新]]
- [[课程/017-value-propagation|价值传播]]
- [[课程/028-train-frozen-lake-q-table|FrozenLake Q 表训练]]
- [[概念/Q表完整训练流程与公式|带公式与 PlantUML 总图的完整 Q 表训练流程]]

> [!success] 已验证实例
> FrozenLake 练习中，一条终止经验把观察 14、动作 RIGHT 对应的 Q 值从 0 更新并写回为 0.2；完整训练后的冻结评估达到 20/20。
