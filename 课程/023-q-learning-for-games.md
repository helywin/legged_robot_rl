---
title: Q-learning 能不能用来玩游戏
aliases:
  - Q-learning 玩游戏
  - 超级玛丽强化学习问题
tags:
  - game-rl/basics
  - quadruped-rl/learning
status: learned
created: 2026-08-28
updated: 2026-08-28
related:
  - "[[021-stage-one-q-learning-review]]"
  - "[[024-dqn-replaces-q-table]]"
  - "[[目标/Chromium-BSU-DQN通关]]"
  - "[[学习主页]]"
---

# Q-learning 能不能用来玩游戏

## 本课问题

普通 Q-learning 能不能训练智能体自动玩《超级玛丽》一类游戏？

答案是：**可以用于状态和动作数量较少的简化游戏，但原始游戏画面通常无法用有限 Q 表直接保存。**

## 把游戏对应到强化学习闭环

以简化版横版游戏为例：

- **观察**：角色位置、前方是否有坑、敌人距离；
- **动作**：左、右、跳跃、右跳；
- **奖励**：前进、获得物品、死亡或过关产生不同反馈；
- **episode**：从关卡开始，到死亡或过关结束。

当观察只有少量离散情况时，可以建立 Q 表。例如在“前方有坑”这一行比较前进和跳跃的 Q 值。

## 原始画面的困难

真实游戏画面包含大量像素。只要角色或敌人移动一点，像素组合就会变化，Q 表便需要另一行。即使两个画面非常相似，普通 Q 表也不会自动共享两行之间的经验。

因此原始画面通常需要用神经网络估计 Q 值，进入 [[024-dqn-replaces-q-table|DQN 怎样替代 Q 表]]。

## 学习结论

Q-learning 的闭环可以用于自动游戏；限制来自 Q 表无法枚举大量画面，不是来自游戏不能提供观察、动作和奖励。

## 关联

- 前置：[[021-stage-one-q-learning-review]]
- 下一课：[[024-dqn-replaces-q-table]]
- 游戏目标：[[目标/Chromium-BSU-DQN通关]]
- 上级：[[学习主页]]
