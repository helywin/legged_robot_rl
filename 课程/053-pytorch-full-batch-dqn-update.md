---
title: 把在线支路和目标支路合成完整批量DQN更新
aliases:
  - PyTorch 完整批量 DQN 更新
  - Full batch DQN update
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: completed
created: 2026-09-02
updated: 2026-09-02
related:
  - "[[051-pytorch-batch-loss-update]]"
  - "[[052-pytorch-batch-dqn-targets]]"
  - "[[概念/DQN训练流程]]"
  - "[[概念/DQN完整训练流程与公式]]"
  - "[[概念/批量损失与梯度平均]]"
  - "[[概念/终止掩码与批量DQN目标]]"
  - "[[概念/目标网络]]"
  - "[[学习主页]]"
  - "[[054-replay-samples-to-tensors]]"
---

# 把在线支路和目标支路合成完整批量 DQN 更新

## 本课不增加新公式

前两课已经分别完成：

- 在线支路：旧观察整批前向，逐行取得实际动作 Q 值，再计算平均损失；
- 目标支路：下一观察整批进入目标网络，用终止掩码得到无梯度 targets。

本课只回答：

> 同一批经验怎样同时流入两条支路，在 loss 处汇合，并且只更新在线网络？

这就是完整 DQN 中一次**批量参数更新**的内部结构，但还不包含环境循环、经验回放抽样和目标网络定期同步。

## 1. 同一批经验被拆成两种用途

固定批量包含两条经验：

| 经验 | 旧观察 $s$ | 动作 $a$ | 奖励 $r$ | 下一观察 $s'$ | `terminated` |
| ---: | --- | ---: | ---: | --- | --- |
| 0 | `[0.2, -0.1]` | 1 | 0.20 | `[0.4, 0.2]` | `False` |
| 1 | `[-0.3, 0.4]` | 0 | 1.00 | `[-0.2, 0.3]` | `True` |

旧观察和动作进入在线支路，回答“在线网络当时怎样评价实际动作”；奖励、下一观察和终止标记进入目标支路，回答“这次经验给出的学习参照是多少”。

## 2. 在线支路产生可训练的预测

旧观察整批进入在线网络：

```text
online_q_values = [
    [0.10, -0.05],
    [0.60,  0.70],
]
```

根据历史动作 `[1,0]` 逐行选取：

```text
selected_q_values = [-0.05, 0.60]
```

这两个值由在线参数计算得到，所以保留计算图：

```text
在线参数 → online_q_values → selected_q_values
```

以后创建的 loss 能沿这条路径找到在线参数。

## 3. 目标支路产生固定参照

下一观察整批进入目标网络：

```text
next_q_values = [
    [0.90, -0.10],
    [0.50,  0.55],
]
best_next_q = [0.90, 0.55]
```

终止标记 `[False, True]` 变成未来掩码 `[1,0]`：

```text
target_q_values = [
    0.20 + 0.9 × 0.90 × 1,
    1.00 + 0.9 × 0.55 × 0,
]
= [1.01, 1.00]
```

整个目标支路位于 `torch.no_grad()` 中：

```text
目标参数 → next_q_values → target_q_values  这里不建立反向路径
```

target 虽然也是本轮刚计算出来的，但它只作为固定参照，不把梯度传回目标网络。

## 4. 两条支路在 loss 处汇合

逐条平方损失：

```text
第0条：(-0.05 - 1.01)² = 1.1236
第1条：( 0.60 - 1.00)² = 0.1600

per_item_losses = [1.1236, 0.1600]
loss = mean(per_item_losses) = 0.6418
```

完整依赖关系是：

```text
在线参数 → 在线预测 → selected_q ─┐
                                     ├→ 逐条损失 → mean loss
目标参数 → 无梯度 target ───────────┘
```

`loss.backward()` 只能沿仍有计算图的在线支路返回，因此把梯度写进在线参数 `.grad`，目标参数 `.grad` 仍为 `None`。

## 5. loss 数字相同，不代表更新方向相同

第 51 课第二条经验使用：

```text
prediction = 0.60
target = 0.20
平方损失 = 0.16
```

本课第二条经验使用：

```text
prediction = 0.60
target = 1.00
平方损失 = 0.16
```

两个 loss 数字都等于 `0.16`，但方向正好相反：

- target 为 `0.20` 时，预测应该减小；
- target 为 `1.00` 时，预测应该增大。

平方损失显示“离目标多远”；梯度通过平方之前保留的有符号差值，指出“应该向哪边移动”。因此只盯着 loss 数字，看不出参数更新方向。

## 6. optimizer 为什么只更新在线网络

optimizer 创建时只接收在线参数：

```text
optimizer 管理：在线参数引用
optimizer 不管理：目标参数
```

本轮接力是：

```text
loss.backward()
→ 在线参数.grad 得到整批梯度
→ 目标参数.grad 仍为 None
→ optimizer.step()
→ 只读取并修改它管理的在线参数
```

optimizer 仍不需要绑定 loss。loss 和 optimizer 通过同一批在线参数对象的 `.grad` 接力。

## 7. 用新参数重新前向会看到什么

学习率为 `0.1`，一次更新后，同一批旧观察的所选动作预测变成：

```text
selected_q_before = [-0.05, 0.60]
target_q_values   = [ 1.01, 1.00]
selected_q_after  = [ 0.0613, 0.65]
```

两项都向各自 target 增大，平均损失从：

```text
0.6418 → 0.5113
```

发生变化的是在线参数和由它们产生的新预测；保持不变的是目标网络参数、target 数值和这批历史经验本身。

## 8. 一次完整批量更新的顺序

```text
清空在线参数旧梯度
→ 旧观察进入在线网络
→ 逐行取得实际动作预测
→ 下一观察进入无梯度目标网络
→ 奖励、最大未来值和终止掩码组成 targets
→ 逐条平方损失
→ 取平均得到标量 loss
→ backward 写入在线参数.grad
→ optimizer.step 只更新一次在线参数
```

这里的顺序不是 API 口诀，而是数据依赖：没有在线预测和 target 就没有 loss，没有 loss 就不能计算梯度，没有 `.grad`，optimizer 就不知道参数应怎样改变。

## 本课训练

打开：

- `exercises/pytorch_full_batch_dqn_update.py`

只修改 `full_batch_dqn_update()` 中的 TODO，把第 51、52 课已经分别完成的两条支路拼起来。

运行：

```bash
.venv/bin/python -m exercises.pytorch_full_batch_dqn_update
```

通过时会确认在线预测、targets、逐条损失、平均 loss、更新方向，以及在线网络改变而目标网络保持不变。

## 当前边界

> [!success] 学习者练习结果
> 学习者把在线批量预测、逐行动作索引、无梯度目标网络、终止掩码、平均损失、反向计算和 optimizer 接成一次完整更新。实际结果为 `selected=[-0.05,0.60]`、`targets=[1.01,1.00]`、`loss=0.6418`；更新后 selected 变为 `[0.0613,0.65]` 且 loss 下降，只有在线参数改变。

> [!warning] 尚未验证
> 当前没有经验回放随机抽样、目标网络定期同步、多层网络、CartPole 完整训练、检查点或独立评估。

## 一句话总结

完整批量 DQN 更新让旧观察沿在线支路保留梯度，让奖励和下一观察沿目标支路提供无梯度参照，两条支路在平均 loss 汇合，最后只有在线参数被 optimizer 修改。

## 关联

- 在线批量损失：[[051-pytorch-batch-loss-update]]
- 批量目标：[[052-pytorch-batch-dqn-targets]]
- 简要流程：[[概念/DQN训练流程]]
- 完整总图：[[概念/DQN完整训练流程与公式]]
- 下一课：[[054-replay-samples-to-tensors|回放缓冲区样本怎样组装成批量张量]]
