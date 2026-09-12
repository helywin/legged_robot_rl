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
updated: 2026-09-12
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

## 三行经验，逐行计算

假设奖励都为 1，下一状态最大 Q 都为 4，γ = 0.9；区别仅在停止原因：

| 经验 | terminated | truncated | future_mask | target |
| --- | --- | --- | --- | --- |
| 继续运行 | False | False | 1 | 1 + 0.9 × 4 = 4.6 |
| 真正结束 | True | False | 0 | 1 + 0 = 1 |
| 外部时间截断 | False | True | 1 | 1 + 0.9 × 4 = 4.6 |

把三行放成张量，只是同时完成上面三次计算。奖励不因终止而删除；开关只乘未来项。第三行必须使用截断前的最后观察，不能使用重置后的第一帧。

代码还需保证奖励、下一 Q 和掩码按行对应、形状一致。掩码也不是修复非法数值的办法：浮点计算中 `0 × NaN` 仍是 NaN；若终止状态不适合送入网络，应只计算合法的非终止行并组装目标。参见 [[概念/PyTorch张量]]。

## 职责边界

- 掩码只控制未来项，不修改奖励；
- 真正终止行的未来项乘 0，target 只剩奖励；
- 截断不等于真正终止，不能自动把未来掩码设为 0；
- 整个目标支路不保留梯度，目标网络参数保持不变。

## 对应课程与代码

- [[052-pytorch-batch-dqn-targets|一批经验怎样分别处理终止与未来价值]]
- `exercises/pytorch_batch_dqn_targets.py`

学习者在第 052 课完成了批量 target 练习：未终止行保留未来最大 Q 值，终止行通过 0 掩码只使用奖励，目标网络没有梯度或参数变化。该课的练习范围止于目标构造；完整连接见 [[053-pytorch-full-batch-dqn-update]]。
