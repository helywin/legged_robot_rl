---
title: DQN训练流程
aliases:
  - DQN 单步更新
  - DQN training flow
tags:
  - knowledge-graph/core
  - reinforcement-learning/dqn
status: learning
created: 2026-09-01
updated: 2026-09-02
related:
  - "[[概念/DQN与神经网络估值]]"
  - "[[概念/DQN完整训练流程与公式]]"
  - "[[概念/经验回放]]"
  - "[[概念/目标网络]]"
  - "[[概念/预测误差与参数更新]]"
  - "[[概念/训练与评估]]"
  - "[[概念/DQN目标Q值]]"
  - "[[概念/批量张量与逐行动作索引]]"
---

# DQN 训练流程

一条回放经验进入 DQN 更新时，在线网络负责当前预测，目标网络负责未来估计，二者形成误差后只直接更新在线网络。

```text
经验回放抽样
  → 在线网络计算所选动作的预测 Q 值
  → 目标网络计算下一观察的未来 Q 值
  → 奖励与未来价值组成目标 Q 值
  → 目标减预测得到误差
  → 更新在线网络
  → 按固定间隔同步目标网络
```

## 关键边界

- 一条经验直接评价当时实际执行的动作；
- 非终止经验使用目标网络的最大未来 Q 值；
- 真正终止时目标只使用奖励；
- 截断不等于任务状态终止，仍可保留未来价值；
- 训练更新在线参数，冻结评估不更新参数；
- 目标网络只在同步时复制在线参数。

## 对应课程与代码

- 带公式和 PlantUML 总图：[[概念/DQN完整训练流程与公式]]
- [[课程/035-one-dqn-training-step|一条回放经验怎样完成一次 DQN 更新]]
- `examples/dqn_training_flow_demo.py`
- PyTorch target：[[048-pytorch-dqn-target-q|奖励和下一观察怎样形成 DQN 目标值]]
- PyTorch 单条经验更新：[[049-pytorch-one-dqn-update|把预测、target 和 optimizer 合成一次 DQN 更新]]
- PyTorch 批量逐行索引：[[050-pytorch-batch-selected-q|一批经验怎样逐行取得实际动作 Q 值]]

PyTorch 实现已经把单条经验的两条支路实际拼接：在线网络的 `selected_q` 保留梯度，目标网络的 `target_q` 不保留梯度。`loss` 通过本轮计算图连到在线参数，`backward()` 把梯度写入这些参数的 `.grad`，只管理在线参数的 optimizer 再读取 `.grad` 并修改在线参数。optimizer 不需要也不会直接绑定 loss。

> [!warning] 当前边界
> 当前已验证单条经验和无隐藏层网络的数据流，正在学习批量索引；还没有批量损失、多层神经网络、完整 episode 训练、检查点或独立评估。
