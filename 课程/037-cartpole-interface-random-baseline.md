---
title: CartPole 环境接口与随机策略基线
aliases:
  - CartPole 随机基线
  - CartPole reset step
tags:
  - reinforcement-learning/environment
  - reinforcement-learning/evaluation
  - game-rl/cartpole
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[036-choose-cartpole-dqn-benchmark]]"
  - "[[概念/标准环境接口]]"
  - "[[概念/训练与评估]]"
  - "[[概念/DQN小型基准]]"
  - "[[学习主页]]"
---

# CartPole 环境接口与随机策略基线

## 本课目标

长期方向仍是用 DQN 学习 Chromium B.S.U.，之后再沿另一条路线进入 Go2。本课不训练 DQN，只确认 CartPole 的真实环境接口，并测量一个不学习的随机策略能坚持多久。

## 为什么先测随机策略

随机策略像一个闭着眼睛、每一步抛硬币决定向左还是向右的人。它没有 Q 值，没有神经网络，也不会根据结果修改自己。

先记录它的表现，相当于在尺子上标出零点。以后 DQN 的奖励升高时，才能比较它是否真的超过了“碰巧乱按”。

## 一次真实接口调用

运行：

```bash
.venv/bin/python -m examples.cartpole_random_baseline
```

固定种子 `20260901` 的第一步实际输出：

```text
reset_observation=[0.013218, 0.008241, -0.014578, -0.020939]
reset_info={}
action=1 (向右推)
step_result=observation=[0.013383, 0.203569, -0.014996, -0.318186], reward=1.0, terminated=False, truncated=False, info={}
```

`reset()` 返回的四个观察值依次是：

1. 小车位置；
2. 小车速度；
3. 杆的角度；
4. 杆的角速度。

执行动作 `1` 后，`step()` 返回五项：

```text
新观察, 奖励, terminated, truncated, info
```

本步奖励为 `1.0`，两个结束标志都是 `False`，说明这一轮还可以继续。`terminated=True` 表示杆倒下或小车越界；坚持到 500 步上限则由时间限制报告 `truncated=True`。

## 随机策略基线

本次唯一运行条件：

- 任务：`CartPole-v1`；
- 策略：每一步从左右动作中随机采样；
- 回合数：100；
- 基础随机种子：`20260901`，第 N 轮使用基础种子加 N；
- 不使用 Q 值、神经网络、经验回放或参数更新。

实际统计：

```text
mean_return=21.40
shortest_episode=9
longest_episode=57
terminated_episodes=100
truncated_episodes=0
```

CartPole 每坚持一步就得到 `+1`，所以平均回报 `21.40` 也表示随机策略平均只维持约 21 步。100 轮全部自然失败，没有一轮达到 500 步时间上限。

这不是“随机策略必须永远等于 21.40”。更换随机种子或回合数会让统计略有变化；固定种子的作用是让当前基线可以重复检查。

## 当前证据边界

> [!success] 启动检查
> Gymnasium 1.3.0 中的 `CartPole-v1` 已真实启动，`reset/step` 五元组与终止语义已由运行结果确认；100 回合随机基线可用同一命令重复得到相同统计，相关测试通过。

> [!warning] 尚未验证
> 当前没有建立神经网络、训练 DQN、保存检查点或进行 GUI 策略回放，也没有验证 Chromium B.S.U.、Isaac Lab、Go2 仿真或真机。

## 一句话总结

随机策略平均只能维持约 21 步，它为后续判断 DQN 是否真正学会平衡提供了可重复的对照起点。

## 关联

- 前置：[[036-choose-cartpole-dqn-benchmark|选择 CartPole 作为第一个 DQN 基准]]
- 环境接口：[[概念/标准环境接口]]
- 评估方法：[[概念/训练与评估]]
- 基准边界：[[概念/DQN小型基准]]
- 实验记录：[[experiments/2026-09-01-cartpole-random-baseline/README|CartPole 随机策略基线实验]]
- 学习入口：[[学习主页]]

