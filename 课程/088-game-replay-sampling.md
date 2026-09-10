---
title: 把采样批次接入经验回放保存与抽样
status: completed
created: 2026-09-10
tags:
  - reinforcement-learning/replay
related:
  - "[[087-real-game-sampling-batches]]"
  - "[[033-experience-replay]]"
  - "[[概念/游戏内部状态与策略接口]]"
---

# 把采样批次接入经验回放保存与抽样

087学习者实测通过，用户要求继续下一步。本课复用已学经验回放概念，接通当前SamplingBatch和17项Transition。不提前引入网络更新。

## 三个数量不是一回事

采样批次大小决定每次从游戏收多少条；缓冲区容量决定最多保留多少旧经验；抽样大小决定本次随机取多少条供后续学习。三者可以不同。缓冲区由外层建立一次，多次store_and_sample调用复用，不随batch_limit或terminated清空。

## 同一组数据走到底

以六条经验A到F举例。每条都有完整的观察、动作、奖励、下一观察和终止标记，用动作前x观察0.0到0.5区分，编号仅作说明，不加进网络输入。容量4，抽样大小2。

第一次只收到A，先保存A，当前仅1条不足抽2，返回空列表。不是丢掉A，也不是游戏结束，只是本次还不能抽足。

下一批收到B/C/D/E/F，依序加入：A/B；A/B/C；A/B/C/D；加入E淘汰A，剩B/C/D/E；加入F淘汰B，剩C/D/E/F。最终保存4条。容量是按经验条数，不是按采样批次数。

第一次有效抽样使用random.Random(7)，从C/D/E/F抽2条得到E/C；样本顺序可以与采样时间不同。对应x观察[0.4,0.2]，Python标准库索引抽样已核对。抽样后缓冲区仍是C/D/E/F，后续允许再次抽到同一条，但一次无放回抽样不会重复同一存储条目。

## 哪些信息不能打乱

随机打乱的是完整经验之间的顺序。E的observation/action/reward/next_observation/terminated必须一起走，不能独立抽各字段。连续采样时相邻经验首尾相接；回放抽样后不同经验不再要求相邻，每条内部关系仍保持。

最后一步terminated=True的经验也保留，因为包含最终动作反馈。stop_reason只说明采样器停止原因，不表示要清空旧数据。缓冲区装满后的时间淘汰与回合结束是两件事。

## 对应Python接口

store_and_sample接收已有buffer、本批SamplingBatch、sample_size与外层rng。遍历batch.transitions，每次buffer.add(row)保存一整条；全部加入后看len(buffer)。不足sample_size返回[]；足够则调用buffer.sample(sample_size,rng)并返回其列表。不在函数内重建缓冲区或重设随机种子。

题目提供GameReplayBuffer，采用033课相同deque(maxlen)自动淘汰和random.sample机制，但类型对应当前Transition，避免导入055的torch依赖。存储机制由教师提供，学习者负责批次到回放的实际连接。未套用旧033的标量观察/字符串动作数据类。

## 亲手练习

文件：`exercises/game_replay_sampling.py`。只补全store_and_sample；缓冲区、样本、返回类型和检查都已提供，完整说明在文件开头。

```bash
.venv/bin/python -m exercises.game_replay_sampling
```

关键成功条件：不足时保存但不抽样；跨批复用；容量4时只保留最近4条；抽样不删除；终止经验保留且五个字段不拆开。该函数可消费087产生的真实SamplingBatch，本轮先离线验证连接，未运行原生回放采集或优化器。

## 证据与下一步

本轮读取033/055既有缓冲区接口，运行标准库抽样对照与未完成入口。学习者代码待提供，learning；没有模型训练，抽到样本不等于更新参数。下一概念：对真实游戏采样批次执行保存和抽样，然后检查抽样经验的内容。

2026-09-10学习者实现store_and_sample并通过实际离线检查：首批保存但不抽样、容量4淘汰旧经验、抽样不删除、完整对象保留。函数内sample_size<=0的后续判断因入口验证而冗余，但不影响结果，不强制重写。088完成，进入[[089-real-game-replay-capacity]]。
