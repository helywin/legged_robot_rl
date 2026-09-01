---
title: 所选动作的 Q 值预测误差
aliases:
  - Q 值预测误差
  - 目标值与预测值的差距
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/dqn
  - reinforcement-learning/python
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[030-predict-action-q-values]]"
  - "[[032-update-one-shared-parameter]]"
  - "[[概念/预测误差与参数更新]]"
  - "[[概念/Q-learning更新]]"
  - "[[学习主页]]"
---

# 所选动作的 Q 值预测误差

## 本课目标

本课只理解“误差”：**目标 Q 值减去刚才所选动作的预测 Q 值，用来判断预测偏高还是偏低。**

## 一条经验提供了什么

假设估计器输出：

```text
LEFT  = 0.20
RIGHT = 0.70
```

策略实际执行了 `RIGHT`。环境返回奖励和新观察后，沿用之前学过的 Q-learning 目标计算，得到：

```text
RIGHT 的目标 Q 值 = 1.00
```

这条经验的误差是：

```python
error = target_q - predicted_q
error = 1.00 - 0.70  # +0.30
```

- 误差大于 0：预测偏低；
- 误差小于 0：预测偏高；
- 误差等于 0：预测等于当前目标。

## 为什么不同时纠正 LEFT

本步真正执行的是 `RIGHT`，环境反馈只能直接说明“这个观察下执行 RIGHT 后发生了什么”。没有执行 `LEFT`，就没有这条反事实经验。因此，本条经验直接比较的是所选动作的预测值，而不是把全部动作输出都与同一个目标比较。

## 与 Q 表的联系

之前的 Q-learning 更新中：

```python
new_q = old_q + learning_rate * (target_q - old_q)
```

括号中的 `target_q - old_q` 就是同一种误差信号。区别是 Q 表将它用于一个格子，参数估计器将它用于共享参数。

> [!success] 本课结论
> 一条经验直接评价的是本次所选动作。目标值不是奖励本身；它仍按 Q-learning 规则结合奖励、下一观察价值和真正终止状态计算。

## 证据边界

`tests/test_single_parameter_q_update.py` 已检查正误差、负误差和所选预测的计算含义。这里仍没有训练完整网络。

## 关联

- 前置：[[030-predict-action-q-values|根据观察预测每个动作的 Q 值]]
- 下一课：[[032-update-one-shared-parameter|根据误差更新一个共享参数]]
- 原有更新：[[概念/Q-learning更新]]
- 概念：[[概念/预测误差与参数更新]]
- 上级：[[学习主页]]
