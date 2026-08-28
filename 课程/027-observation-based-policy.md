---
title: 从固定动作序列到观察策略
aliases:
  - 观察策略编程练习
  - 可重复使用的手写策略
tags:
  - game-rl/tabular-q
  - reinforcement-learning/policy
  - reinforcement-learning/python
status: completed
created: 2026-08-28
updated: 2026-08-28
related:
  - "[[026-python-frozen-lake-environment]]"
  - "[[005-policy]]"
  - "[[学习主页]]"
  - "[[教学计划]]"
---

# 从固定动作序列到观察策略

## 本课只解决一个问题

上一题按照调用次数依次返回动作，第一局可以到达终点，但第二局会因为全局下标没有复位而报错。本课把动作的依据改成当前观察：角色处于哪个格子，就返回该格子对应的动作。

## 学习者实现

练习代码位于 `exercises/frozen_lake_observation_policy.py`。学习者删除了模板中错误的 `del observation`，并使用 `(观察编号, 动作)` 对组成手写规则表，再根据传入的 `observation` 查找动作。

实际使用的安全路径是：

```text
0 → 1 → 2 → 6 → 10 → 14 → 15
```

每个关键观察对应的动作是：

| 当前观察 | 动作 | 下一个观察 |
| ---: | --- | ---: |
| 0 | RIGHT | 1 |
| 1 | RIGHT | 2 |
| 2 | DOWN | 6 |
| 6 | DOWN | 10 |
| 10 | DOWN | 14 |
| 14 | RIGHT | 15 |

## 实际验证

运行命令：

```bash
.venv/bin/python -m exercises.frozen_lake_observation_policy
```

同一 Python 进程连续运行两局，两局都在第 6 步到达终点并获得 `+1` 奖励，最终输出：

```text
练习通过：观察策略在 reset() 后仍可重复使用
```

随后运行全量测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：35 项测试全部通过。

> [!success] 今天确认的概念
> 策略可以被看成“观察到动作”的对应关系。只要同一观察得到同一决策，它就不需要记住自己是第几次被调用，环境 `reset()` 后也可以直接复用。

## 它为什么还不是 Q 表

当前 `action_table` 是人写出的规则：每个观察只保存最终选择的动作。Q 表则会为每个“观察—动作”组合保存一个可更新的分数，并由训练过程通过尝试和奖励学出选择。

```text
当前手写规则：观察 → 人指定的动作
Q 表策略：    观察 → 比较该行所有动作的 Q 值 → 选择动作
```

> [!warning] 证据边界
> 本课验证的是纯 Python 手写观察策略，不是 Q-learning 训练；尚未验证自动探索、Q 值更新、Gymnasium、DQN、Chromium B.S.U.、Isaac Lab 或真机。

## 下一步

下一课只加入一项变化：保留同一个 FrozenLake 环境，让程序从零更新 `16×4` Q 表并学出路线。

## 关联

- 环境：[[026-python-frozen-lake-environment]]
- 策略概念：[[005-policy]]
- 上级：[[学习主页]]
- 路线：[[教学计划]]
