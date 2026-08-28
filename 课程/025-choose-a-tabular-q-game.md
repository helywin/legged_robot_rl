---
title: 选择第一个表格 Q-learning 游戏
aliases:
  - FrozenLake 选择
  - 表格 Q-learning 游戏
tags:
  - game-rl/tabular-q
  - reinforcement-learning/environment
status: learned
created: 2026-08-28
updated: 2026-08-28
related:
  - "[[022-standard-env-interface]]"
  - "[[024-dqn-replaces-q-table]]"
  - "[[目标/Chromium-BSU-DQN通关]]"
  - "[[026-python-frozen-lake-environment]]"
  - "[[学习主页]]"
---

# 选择第一个表格 Q-learning 游戏

## 为什么选择 FrozenLake

第一个练习选择 FrozenLake（冰湖），因为它既像一个小型游戏，又能使用有限 Q 表：

```text
S F F F
F H F H
F F F H
H F F G
```

- `S` 是起点；
- `F` 是安全冰面；
- `H` 是失败的冰洞；
- `G` 是终点；
- 动作是上、下、左、右。

4×4 地图有 16 个离散观察和 4 个离散动作，所以 Q 表是 `16×4`，可以完整查看。

## 第一次实践的唯一变化

先使用不打滑的确定性版本。这样执行哪个动作就朝哪个方向移动，只观察“随机策略”变成“训练后的 Q 表策略”这一项变化。

第一次实践将比较：

1. 未训练的随机策略；
2. 使用 ε-greedy 训练的 Q 表；
3. 关闭探索并冻结 Q 表后的独立评估；
4. 在终端逐步回放学到的路线。

## 当前环境状态

2026-08-28 检查时，系统 Python 有 NumPy，但没有 Gymnasium。为了不把安装依赖混进第一次游戏实验，下一课先实现一个只依赖 Python 标准库、接口形状接近 Gymnasium 的 FrozenLake。

这仍然只是表格 Q-learning 小游戏，不是 DQN，也没有验证 Chromium B.S.U.。

## 关联

- 环境接口：[[022-standard-env-interface]]
- DQN 动机：[[024-dqn-replaces-q-table]]
- 游戏目标：[[目标/Chromium-BSU-DQN通关]]
- 下一课：[[026-python-frozen-lake-environment|纯 Python FrozenLake 环境与随机基线]]
- 上级：[[学习主页]]
