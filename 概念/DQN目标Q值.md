---
title: DQN目标Q值
aliases:
  - DQN target Q
  - 奖励与未来价值目标
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: learned
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[概念/所选动作Q值与索引]]"
  - "[[概念/目标网络]]"
  - "[[概念/标准环境接口]]"
  - "[[概念/DQN训练流程]]"
  - "[[048-pytorch-dqn-target-q]]"
  - "[[049-pytorch-one-dqn-update]]"
---

# DQN 目标 Q 值

DQN 用一条经验中的奖励和下一观察估计值，构造当前所选动作应该靠近的目标。

```text
非终止：target = reward + 折扣因子 × 目标网络的最大下一 Q 值
真正终止：target = reward
```

下一 Q 值来自目标网络，不是当前正在更新的在线网络。计算 target 时关闭梯度，使 `backward()` 只沿在线网络的 `selected_q` 一侧传播。

## 结束语义

- `terminated=True`：任务状态真正结束，没有未来价值；
- `truncated=True`：只是时间或步数上限，任务状态本身未终止，仍可保留下一观察的未来价值；
- 两者都会结束当前 episode，但不能因此使用相同的 target。

## 职责边界

- target 不是即时奖励的同义词，非终止时还包含折扣后的未来估计；
- 最大下一 Q 值用于构造学习目标，不等于环境下一步真的执行该动作；
- target 不需要梯度，目标网络也不在在线优化器的这次更新中改变；
- 本节点只构造 target，还没有把它与 `selected_q` 组成完整更新。

## 对应课程与代码

- [[048-pytorch-dqn-target-q|奖励和下一观察怎样形成 DQN 目标值]]
- `examples/pytorch_dqn_target_q.py`
- `exercises/pytorch_dqn_target_q.py`

学习者已经完成实际练习：非终止目标为 `1.01`，真正终止目标为 `0.20`，两个目标都不需要梯度，目标网络参数没有梯度且保持不变。
