---
title: 终止掩码与批量DQN目标
aliases:
  - Termination mask
  - 批量 DQN target
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: learned
created: 2026-09-02
updated: 2026-09-02
related:
  - "[[概念/DQN目标Q值]]"
  - "[[概念/目标网络]]"
  - "[[概念/标准环境接口]]"
  - "[[概念/DQN训练流程]]"
  - "[[052-pytorch-batch-dqn-targets]]"
  - "[[053-pytorch-full-batch-dqn-update]]"
---

# 终止掩码与批量 DQN 目标

批量经验中的每一行可能具有不同的 `terminated`。未来掩码把未终止行转换为 1、真正终止行转换为 0，使统一的张量公式能逐行决定是否保留未来价值：

$$
target=r+\gamma\times best\_next\_q\times future\_mask
$$

## 职责边界

- 掩码只控制未来项，不修改奖励；
- 真正终止行的未来项乘 0，target 只剩奖励；
- 截断不等于真正终止，不能自动把未来掩码设为 0；
- 整个目标支路不保留梯度，目标网络参数保持不变。

## 对应课程与代码

- [[052-pytorch-batch-dqn-targets|一批经验怎样分别处理终止与未来价值]]
- `exercises/pytorch_batch_dqn_targets.py`

学习者已经完成批量 target 练习：未终止行保留未来最大 Q 值，终止行通过 0 掩码只使用奖励，目标网络没有梯度或参数变化。当前尚未与在线批量损失接成一次完整更新。
