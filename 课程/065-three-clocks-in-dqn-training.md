---
title: DQN训练为什么有三种计数
aliases:
  - DQN three clocks
  - 环境步更新次数与回合数
tags:
  - reinforcement-learning/dqn
  - reinforcement-learning/training
  - reinforcement-learning/python
status: completed
created: 2026-09-04
updated: 2026-09-04
related:
  - "[[064-mean-loss-and-single-sample-tradeoff]]"
  - "[[概念/DQN训练的三种计数]]"
  - "[[概念/回放预填充与训练起点]]"
  - "[[概念/目标网络]]"
  - "[[概念/DQN完整训练流程与公式]]"
  - "[[学习主页]]"
---

# DQN 训练为什么有三种计数

## 本课只解决一个混乱

完整 DQN 程序里经常同时出现：

```text
environment_step
update_count
episode_count
```

它们看起来都在“循环里加一”，但记录的是三种不同事件。如果把它们混为一个数字，就会弄不清：预填充阶段到底有没有工作、什么时候同步目标网络、一次 episode 做了多少次训练。

本课只解释三条时间线，不增加网络公式，也不创建代码练习。

## 1. 先用驾校的三个计数类比

想象学车时同时记录：

- 汽车一共行驶了多少米；
- 教练一共纠正了多少次方向盘；
- 一共完成了多少次上车练习。

车可以先行驶一段，教练才第一次纠正；一次上车可以行驶很多米；同一段行驶后，教练也可能连续纠正几次。

DQN 对应关系是：

| DQN 计数 | 它数的事件 | 类比 |
| --- | --- | --- |
| `environment_step` | 环境执行了一次动作 | 又行驶了一段 |
| `update_count` | optimizer 修改了一次在线参数 | 教练完成一次纠正 |
| `episode_count` | 一局结束并重新开始 | 完成一次上车练习 |

它们在同一个程序里发生，但不是同一件事。

## 2. environment_step 数的是数据来源

当前观察为 $s_t$ 时，策略选出动作 $a_t$，环境执行：

```text
next_observation, reward, terminated, truncated = env.step(action)
```

只要这次 `env.step()` 真正发生，就完成了一个 `environment_step`，并产生一条新经验：

```text
(observation, action, reward, next_observation, terminated)
```

无论网络有没有训练，这个计数都应该增加。预填充期间网络参数不变，但环境仍在产生数据，所以 environment step 一直增加。

它回答的是：

> 智能体实际与环境交互、收集了多少步新数据？

## 3. update_count 数的是参数修改

只有完整发生下面这条链，才算一次 update：

```text
从回放缓冲区抽一批经验
→ 在线网络前向
→ 目标网络构造 target
→ 得到平均 loss
→ backward() 写入在线参数 .grad
→ optimizer.step() 修改在线参数
→ update_count += 1
```

只调用 `backward()` 还不能让 update count 增加，因为参数尚未改变。只做环境 `step()` 也不能增加它，因为那只是产生数据。

它回答的是：

> 在线网络参数实际被 optimizer 修改了多少次？

目标网络同步通常依据 `update_count`，原因正是同步关心在线参数已经变化了多少轮。

## 4. episode_count 数的是完整尝试

一个 episode 从：

```text
observation, info = env.reset()
```

开始，经过多个 environment step，直到：

```text
terminated or truncated
```

为真。此时本局结束，`episode_count` 增加，然后再次 `reset()` 开始下一局。

episode 结束不会把以下内容清零：

- 在线网络已经学到的参数；
- 目标网络参数；
- replay buffer 中允许保留的旧经验；
- 从训练开始累计的 `environment_step` 和 `update_count`。

通常只会重置环境内部状态、当前 observation 和本局回报等“本局变量”。

它回答的是：

> 智能体从 reset 到结束，一共完成了多少次完整尝试？

## 5. 用六个环境步走一遍

固定规则：

```text
warmup_size = 3
达到预填充后：每个 environment step 做 1 次 update
sync_interval = 2 个 update
第 2、5 个 environment step 恰好结束当前 episode
```

每个环境步先产生并保存经验，再判断缓冲区是否达到 3 条。

| environment step | buffer 数量 | 是否更新 | update count | 是否同步目标网络 | 已完成 episode 数 |
| ---: | ---: | --- | ---: | --- | ---: |
| 1 | 1 | 否 | 0 | 否 | 0 |
| 2 | 2 | 否 | 0 | 否 | 1 |
| 3 | 3 | 是 | 1 | 否 | 1 |
| 4 | 4 | 是 | 2 | 是 | 1 |
| 5 | 5 | 是 | 3 | 否 | 2 |
| 6 | 6 | 是 | 4 | 是 | 2 |

从表中可以直接看到：

- 前两个 environment step 已经收集数据，但 `update_count` 仍是 0；
- 第 2 步结束第 1 个 episode，但训练尚未开始；
- 第 4 步时 `environment_step=4`、`update_count=2`，两个数字已经不同；
- 目标网络在第 2、4 次参数更新后同步，对应环境步 4、6；
- episode 长短由环境何时结束决定，与同步间隔没有固定对应关系。

## 6. 为什么目标同步不能直接看 environment_step

假设规定“在线网络每更新 2 次，就同步目标网络”。正确判断是：

```text
update_count % 2 == 0
```

如果错误地使用：

```text
environment_step % 2 == 0
```

那么第 2 个环境步就会触发同步。但此时还在预填充，在线网络一次都没有更新；把完全相同的参数再复制一遍没有训练意义。

更严重的是，以后如果每个环境步做 4 次 update，environment step 只增加 1，在线参数却已经变化 4 次。用 environment step 控制同步会让目标网络保持固定过久。

所以计数必须跟它控制的事件一致：

| 要控制的事情 | 应读取的计数 |
| --- | --- |
| 探索了多少新环境状态 | `environment_step` |
| 在线参数更新和目标同步 | `update_count` |
| 完成了多少局及每局回报 | `episode_count` |

## 7. 一次循环里它们分别在哪里变化

```plantuml
@startuml
top to bottom direction

start
:根据当前 observation 选择动作;
:调用 env.step(action)\nenvironment_step 加 1;
:把新经验加入 replay buffer;

if (buffer 已达到 warmup_size?) then (是)
  :抽取一批经验;
  :loss → backward → optimizer.step;
  :update_count 加 1;
  if (update_count 到达同步间隔?) then (是)
    :在线参数复制给目标网络;
  else (否)
    :目标网络参数保持不变;
  endif
else (否)
  :本步只收集经验;
endif

if (terminated 或 truncated?) then (是)
  :episode_count 加 1;
  :env.reset() 开始下一局;
else (否)
  :next_observation 成为当前 observation;
endif

stop
@enduml
```

图中每个条件的“是/否”只标在分支线上；动作框只写真正发生的动作。

## 8. 三个计数不会互相替代

下面三个说法分别可能同时成立：

```text
environment_step = 10000
update_count = 9000
episode_count = 430
```

普通语言含义是：

- 与环境交互了 10000 步；
- 扣掉早期预填充等阶段后，在线参数实际更新了 9000 次；
- 这些环境步分布在 430 个完整或被截断的 episode 中。

不能只写“训练了 10000 次”。这句话没有说明是采集了 10000 步、更新了 10000 次，还是跑了 10000 局，三者计算成本和学习含义完全不同。

## 9. 代码名字背后的机制

| 常见代码 | 读写的状态 | 是否改变网络参数 |
| --- | --- | --- |
| `env.step(action)` | 推进环境并产生经验 | 否 |
| `replay_buffer.add(...)` | 保存一条经验 | 否 |
| `loss.backward()` | 把梯度写入 `.grad` | 否 |
| `optimizer.step()` | 读取 `.grad`，修改在线参数 | **是** |
| `target.load_state_dict(...)` | 复制在线参数数值 | 改变目标参数，但不是梯度学习 |
| `env.reset()` | 重置环境并给出新观察 | 否 |

这张表也解释了为什么 `training` 不是单指某一个函数：完整训练系统同时包含数据采集、参数学习、目标同步和回合管理，但只有 optimizer update 是梯度意义上的网络学习。

## 本课学习方式

这是进入真实 CartPole 训练前的时间线原理课，不创建抽象代码练习。当前只需要建立一个判断标准：看到任何计数时，先问“它究竟在数哪一种事件”。

> [!success] 学习进度
> 学习者完成原理讲解后选择继续，当前概念记为已理解。

## 当前边界

> [!info]
> 六步时间线是手工设定，用来隔离三个计数的职责。它没有执行真实 CartPole 训练，也没有产生检查点或评估结果。

## 一句话总结

环境步负责产生新经验，更新次数负责记录参数被 optimizer 修改，episode 数负责记录从 reset 到结束的完整尝试；三条时间线发生在同一程序中，但不能混用。

## 关联

- 前置：[[064-mean-loss-and-single-sample-tradeoff|平均 loss 与单条经验权衡]]
- 核心概念：[[概念/DQN训练的三种计数]]
- 预填充：[[概念/回放预填充与训练起点]]
- 目标同步：[[概念/目标网络]]
- 完整流程：[[概念/DQN完整训练流程与公式]]
