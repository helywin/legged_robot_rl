---
title: 强化学习双目标
aliases:
  - 学习目标
  - 双目标
tags:
  - reinforcement-learning/goal
  - roadmap
status: active
created: 2026-08-28
updated: 2026-08-28
related:
  - "[[学习主页]]"
  - "[[教学计划]]"
  - "[[000-stage-goal-blind-stairs]]"
  - "[[000-stage-goal-chromium-bsu-dqn]]"
---

# 强化学习双目标

这套课程现在服务于两个长期目标。两个目标先共享强化学习基础，进入神经网络后再分成游戏和机器人两条路线。

## 目标一：四足机器人盲走爬楼梯

训练四足机器人在不读取相机、激光雷达、深度图或外部地形高度扫描的条件下，仅依赖机身内部可获得或估计的状态，在仿真中上下楼梯。完整定义与验证边界见 [[000-stage-goal-blind-stairs|阶段目标：宇树机器狗盲走爬楼梯]]。

这条路线后期主要面对连续观察和连续关节动作，因此在理解神经网络基础后，重点学习 PPO 和 Isaac Lab。

## 目标二：DQN 打通 Chromium B.S.U.

训练一个以游戏观察为输入、输出离散游戏动作的 DQN 智能体，在无人操作下从新游戏开始并完成 `chromium-bsu`。完整目标与待验证边界见 [[000-stage-goal-chromium-bsu-dqn|阶段目标：DQN 打通 Chromium B.S.U.]]。

这条路线后期主要面对游戏画面和离散动作，因此重点学习 DQN、游戏环境封装和独立回放评估。

## 共享基础

两个目标都需要先掌握同一个闭环：

```text
观察 → 策略选择动作 → 环境推进 → 奖励和新观察 → 更新策略
```

当前不会同时训练两个复杂目标。学习顺序是：

1. 纯 Python Q-learning；
2. 标准环境接口与 FrozenLake；
3. 神经网络怎样替代 Q 表；
4. DQN 小任务；
5. 分别进入 Chromium B.S.U. 游戏路线和 Isaac Lab 四足路线。

> [!warning] 验证边界
> FrozenLake 或 DQN 小任务成功不代表已经能打通 Chromium B.S.U.；Isaac Lab 仿真成功也不代表已经完成真机验证。两个目标始终分别记录证据。

## 关联

- 学习入口：[[学习主页]]
- 完整顺序：[[教学计划]]
- 机器人目标：[[000-stage-goal-blind-stairs]]
- 游戏目标：[[000-stage-goal-chromium-bsu-dqn]]
