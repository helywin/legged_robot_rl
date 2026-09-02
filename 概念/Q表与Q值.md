---
title: Q表与Q值
aliases:
  - Q 表
  - Q value table
tags:
  - knowledge-graph/core
  - reinforcement-learning/q-learning
status: active
related:
  - "[[概念/策略]]"
  - "[[概念/探索与利用]]"
  - "[[概念/Q-learning更新]]"
  - "[[概念/Q表完整训练流程与公式]]"
---

# Q 表与 Q 值

Q 表为每个“观察—动作”组合保存一个估计分数。某一行代表当前观察，各列代表可选动作。

Q 值不是当前位置编号，也不是立即奖励；它估计从这里选择某动作后能够得到的累计价值。

## 对应课程

- [[课程/010-q-table|Q 表如何保存经验]]
- [[课程/014-future-value-and-discount|为什么 Q 值考虑下一位置]]
- [[课程/028-train-frozen-lake-q-table|训练 FrozenLake 的 16×4 Q 表]]
- [[概念/Q表完整训练流程与公式|Q 表从选动作、单格更新到冻结评估的完整流程]]
