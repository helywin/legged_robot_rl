---
title: 一批经验怎样分别处理终止与未来价值
aliases:
  - PyTorch 批量 DQN target
  - DQN termination mask
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: completed
created: 2026-09-02
updated: 2026-09-02
related:
  - "[[051-pytorch-batch-loss-update]]"
  - "[[概念/终止掩码与批量DQN目标]]"
  - "[[概念/DQN目标Q值]]"
  - "[[概念/目标网络]]"
  - "[[概念/标准环境接口]]"
  - "[[概念/DQN训练流程]]"
  - "[[学习主页]]"
  - "[[053-pytorch-full-batch-dqn-update]]"
---

# 一批经验怎样分别处理终止与未来价值

## 本课只解决一个问题

单条经验只有一个 `terminated`，可以直接使用 `if`。一批经验中的结束情况可能不同：

```text
terminated = [False, True]
```

本课只回答：

> 目标网络整批计算下一观察后，怎样让未终止经验保留未来价值，同时让真正终止的经验完全去掉未来价值？

这里新增的工具叫**掩码**（mask）：用与批量等长的 1 和 0，决定每一行的未来价值是否保留。

## 1. 固定的两条经验

| 经验 | reward | next observation | terminated |
| ---: | ---: | --- | --- |
| 0 | 0.20 | `[0.4, 0.2]` | `False` |
| 1 | 1.00 | `[-0.2, 0.3]` | `True` |

折扣因子：

```text
discount_factor = 0.9
```

第 0 条还会继续，因此 target 要包含下一观察的未来估计；第 1 条已经真正结束，因此 target 只能是当前奖励 1.00。

## 2. 目标网络整批前向得到什么

两条下一观察同时进入暂时固定的目标网络：

```text
next_q_values = [
    [0.90, -0.10],
    [0.50,  0.55],
]
```

形状含义：

```text
(经验数量, 动作数量) = (2, 2)
```

每行取最大动作 Q 值：

```text
best_next_q = [0.90, 0.55]
```

注意：第二行虽然也算出了 `0.55`，但这条经验已经终止。这个数字会在下一步被掩码乘成 0，不会进入 target。

## 3. 怎样从 terminated 得到未来掩码

原始布尔值是：

```text
terminated = [False, True]
```

我们需要的含义是“还能不能保留未来价值”：

```text
False（没终止） → 保留未来 → 1
True （已终止） → 去掉未来 → 0
```

因此先对布尔值取反，再转换成浮点数：

```text
future_mask = [1.0, 0.0]
```

掩码不是新的奖励，也不是模型预测；它只是逐行控制未来项的开关。

## 4. 两条 target 分别怎样手算

统一写成：

$$
target=r+\gamma\times best\_next\_q\times future\_mask
$$

### 第 0 条：没有终止

```text
target_0
= 0.20 + 0.9 × 0.90 × 1
= 1.01
```

### 第 1 条：真正终止

```text
target_1
= 1.00 + 0.9 × 0.55 × 0
= 1.00
```

所以整批目标为：

```text
target_q_values = [1.01, 1.00]
```

第 1 条中的 `0.55` 虽然被目标网络计算过，但乘 0 后对 target 没有任何贡献。

## 5. 为什么整个目标支路都不需要梯度

target 是在线预测本轮追赶的参照。本轮 optimizer 只管理在线网络，不能让 target 随同一次反向传播一起变化。

```text
下一观察 → 目标网络 → 每行最大未来 Q 值
                           ↓ 乘终止掩码
奖励 ─────────────────→ target_q_values
```

这条支路整体放在 `torch.no_grad()` 中，因此：

- `target_q_values.requires_grad` 为 `False`；
- 目标网络参数 `.grad` 保持 `None`；
- 目标网络参数值保持不变；
- 后续 loss 的反向路径只从在线预测一侧回到在线参数。

## 6. terminated 与 truncated 仍不能混淆

本课输入只传入真正终止标记 `terminated`：

- 到达任务终点或失败终点：`terminated=True`，未来掩码为 0；
- 只是时间上限截断：`truncated=True` 但 `terminated=False`，未来掩码仍为 1。

回合循环可以在 `terminated or truncated` 时停止，但构造 target 时不能把二者都当作“没有未来”。

## 7. 数据流与状态变化

```text
next_observations (2, 2)
  → 目标网络批量前向
next_q_values (2, 2)
  → 每行取最大值
best_next_q (2,)

terminated (2,)
  → 取反并转浮点数
future_mask (2,)

rewards + gamma × best_next_q × future_mask
  → target_q_values (2,)
```

这一课没有 loss、`backward()` 或 optimizer，所以在线网络和目标网络参数都不会改变。

## 本课训练

打开：

- `exercises/pytorch_batch_dqn_targets.py`

只修改 `calculate_batch_dqn_targets()` 中的 TODO，完成目标网络批量前向、逐行最大值、终止掩码和 target。

运行：

```bash
.venv/bin/python -m exercises.pytorch_batch_dqn_targets
```

成功时会确认两个 target 数值、终止行的未来项为零，以及目标网络没有梯度或参数变化。

## 当前边界

> [!success] 学习者练习结果
> 学习者完成目标网络整批前向、`max(dim=1)` 逐行最大值、终止掩码和无梯度 target。实际得到 `future_mask=[1,0]`、`target_q_values=[1.01,1.00]`，目标网络梯度为 `None` 且参数保持不变。

> [!warning] 尚未验证
> 还没有把批量 target 与在线预测接成同一次更新、经验回放随机抽样、CartPole 训练、检查点或独立评估。

## 一句话总结

批量 target 用终止掩码逐行控制未来项：未终止行为 1，保留折扣后的最大未来 Q 值；真正终止行为 0，target 只剩当前奖励。

## 关联

- 前置：[[051-pytorch-batch-loss-update|一批误差怎样合成一次参数更新]]
- 概念：[[概念/终止掩码与批量DQN目标]]
- 单条 target：[[概念/DQN目标Q值]]
- 目标网络：[[概念/目标网络]]
- 结束语义：[[概念/标准环境接口]]
- 下一课：[[053-pytorch-full-batch-dqn-update|把在线支路和目标支路合成完整批量 DQN 更新]]
