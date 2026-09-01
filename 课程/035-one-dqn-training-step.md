---
title: 一条回放经验怎样完成一次 DQN 更新
aliases:
  - DQN 单步训练流程
  - 一次 DQN 更新
tags:
  - reinforcement-learning/dqn
  - reinforcement-learning/neural-network
  - reinforcement-learning/python
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[034-target-network]]"
  - "[[概念/DQN训练流程]]"
  - "[[概念/经验回放]]"
  - "[[概念/目标网络]]"
  - "[[概念/预测误差与参数更新]]"
  - "[[学习主页]]"
---

# 一条回放经验怎样完成一次 DQN 更新

## 本课目标

长期方向仍是 DQN 游戏和 Go2 强化学习。本课不增加新的算法部件，只把已经学过的在线参数、经验回放、目标参数、预测、目标、误差和更新串成一次完整流程。

## 一次训练更新的顺序

```text
1. 环境产生一条经验并存入回放缓冲区
2. 从缓冲区随机抽取一条旧经验
3. 在线参数计算旧观察下每个动作的 Q 值
4. 取出当时真正执行动作的预测 Q 值
5. 目标参数计算新观察下每个动作的 Q 值
6. 用奖励和最大未来 Q 值组成目标 Q 值
7. 目标值减预测值得到误差
8. 根据误差更新在线参数
9. 目标参数保持不变，到了同步时刻才复制在线参数
```

其中第 6 步仍遵守之前的结束语义：

- `terminated=True`：目标只使用奖励；
- `truncated=True`：任务状态本身没有结束，仍可保留未来价值。

## 最小标准库示例

`examples/dqn_training_flow_demo.py` 使用两个动作和简化线性参数，只完成一条经验的一次更新。

运行：

```bash
.venv/bin/python -m examples.dqn_training_flow_demo
```

实际输出：

```text
buffer_observations=[1, 2, 3]
sampled=(observation=2, action=LEFT, reward=0.1, new_observation=3)
online_q_values={'LEFT': 0.4, 'RIGHT': 0.8}
selected_prediction=0.400000
next_target_q_values={'LEFT': 0.9, 'RIGHT': 1.5}
target_q=1.450000
error=+1.050000
updated_LEFT_weight=0.200000->0.620000
other_online_weight_unchanged=True
target_weights_changed=False
```

## 逐项对应

1. 回放缓冲区中有观察 `1、2、3` 的三条经验；
2. 固定随机种子抽到了观察 `2`、动作 `LEFT` 的经验；
3. 在线参数输出 `LEFT=0.4、RIGHT=0.8`；
4. 当时执行的是 `LEFT`，所以本条经验直接评价 `0.4`，不是 `RIGHT` 的 `0.8`；
5. 目标参数在新观察 `3` 上输出 `LEFT=0.9、RIGHT=1.5`；
6. 这不是终止经验，因此使用最大未来值 `1.5` 组成目标 `1.45`；
7. 误差为 `1.45 - 0.4 = +1.05`，说明 `LEFT` 的预测偏低；
8. 简化示例将 `LEFT` 在线权重从 `0.2` 更新到 `0.62`；
9. `RIGHT` 在线权重未改，目标参数也未改。

## 真实神经网络中的区别

真实 DQN 不会只保存每个动作一个权重。它使用神经网络的许多共享参数，一次误差会通过网络计算各参数应怎样改变。本课只是把数据流串起来，没有实现多层网络、批量更新或自动求导。

> [!success] 当前证据
> 标准库流程示例实际启动成功，新增 6 项测试验证了多动作预测、非终止目标、真正终止、截断未来价值、只更新所选动作的在线参数和目标参数不变。它不是 DQN 冒烟训练，也没有评测游戏、Isaac Lab、仿真机器人或真机。

## 一句话总结

一次 DQN 更新就是从回放经验中取得旧观察和反馈，用在线参数提供预测、目标参数提供未来估计，再用两者的误差更新在线参数。

## 关联

- 前置：[[034-target-network|目标网络：暂时固定训练目标]]
- 流程：[[概念/DQN训练流程]]
- 回放：[[概念/经验回放]]
- 目标：[[概念/目标网络]]
- 更新：[[概念/预测误差与参数更新]]
- 学习入口：[[学习主页]]
