---
title: DQN小型基准
aliases:
  - DQN 基准任务
  - CartPole DQN 基准
tags:
  - reinforcement-learning/dqn
  - reinforcement-learning/evaluation
  - game-rl
status: active
created: 2026-09-01
updated: 2026-09-12
related:
  - "[[概念/DQN训练流程]]"
  - "[[概念/训练与评估]]"
  - "[[概念/标准环境接口]]"
  - "[[概念/游戏强化学习路线]]"
  - "[[036-choose-cartpole-dqn-benchmark]]"
  - "[[066-first-real-cartpole-dqn-smoke-training]]"
---

# DQN 小型基准

## 定义

DQN 小型基准是一个观察、动作、奖励、结束和重置都足够清楚的简单环境，用来检查完整 DQN 训练闭环是否工作。它不是最终应用，而是进入复杂游戏前的可控练习场。

本课程选择 `CartPole-v1`：四个连续观察量输入网络，网络输出向左、向右两个动作的 Q 值。

## 职责边界

- 它验证真实神经网络、经验回放、目标网络、探索和冻结评估能否协作；
- 它不验证图像观察、独立游戏窗口控制、OCR、游戏奖励设计或外部游戏的自动重开；CartPole 自身的回合重置则属于基准闭环；
- 它更不代表 Chromium B.S.U. 已经通关，也不提供 Go2 仿真或真机证据。

## 怎样评估

先记录随机策略基线，再训练并选择检查点。最终关闭探索和参数更新，使用未参与训练及选检查点的随机种子做独立评估，同时记录平均回报、完整 500 步的回合数和 GUI 行为回放。

## 相关概念

- 前置流程：[[概念/DQN训练流程]]
- 环境约定：[[概念/标准环境接口]]
- 证据边界：[[概念/训练与评估]]
- 应用方向：[[概念/游戏强化学习路线]]
- 对应课程：[[036-choose-cartpole-dqn-benchmark]]
- 运行基线：[[037-cartpole-interface-random-baseline]]
- 第一次训练：[[066-first-real-cartpole-dqn-smoke-training]]

## 当前证据

Gymnasium 1.3.0 的真实环境已经启动。固定基础种子 `20260901` 的 100 回合随机策略平均回报为 21.40，最短 9 步、最长 57 步。

教师参考冒烟训练把真实环境、经验回放、`4→64→ReLU→2` 在线/目标网络和冻结评估接通。固定 30,000 环境步后，20 个评估种子的平均回报从训练前 `9.20` 提高到 `253.60`，超过冒烟线 `200`。这是教师参考数据。

[[066-first-real-cartpole-dqn-smoke-training]] 的后续补记还记录了学习者平均回报 253.6 和向右出轨的诊断；[[067-position-reward-and-policy]] 已有进一步对照，并明确最终 100 回合验收未完成、调参告一段落。因此不能继续把早期“学习者运行结果尚待记录”当成当前的全部状态，也不能由这些数据推定所有 GUI、导出和最终验收均已通过。详见 [[概念/CartPole可恢复状态与控制余量]]。
