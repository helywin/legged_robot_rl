---
title: 真实采样经验怎样保存淘汰与随机抽取
status: optional
created: 2026-09-10
tags:
  - reinforcement-learning/experiment
related:
  - "[[088-game-replay-sampling]]"
  - "[[087-real-game-sampling-batches]]"
---

# 真实采样经验怎样保存淘汰与随机抽取

088学习者代码已通过，当前把同一个store_and_sample接到原生采样。实验完整记录见[[experiments/2026-09-10-chromium-replay-capacity/README]]。

## 先预测同一组数量

固定6条真实经验，编号1到6。先采1条、再采5条，每次都请求抽2条。首批只有1条应先保存，返回[]；容量4最终留3/4/5/6，容量6最终留1/2/3/4/5/6。唯一配置变化是缓冲容量，动作、步数、批次和抽样大小不变。

随机抽取改变的是取出的样本组合，不是缓冲区内容，也不改变游戏状态。种子7的第一次有效抽样中，容量4得到5/3，容量6得到3/2；同一个种子面对不同候选集合不保证抽到相同编号。

## 真实一条经验是什么

教师此次抽到的经验5为observation[0]=0.093、action=4、reward=0、next_observation[0]=0.126、terminated=False。它表示真实游戏中的横坐标从约0.93变成1.26，按位置尺度10编码；动作是RIGHT，期间没有新增分数。随机取出它以后，仍然保留这五项对应关系。另一个样本经验3为0.039→0.063，时间上不必紧挨经验5。

采样时相邻经验要衔接，抽样时不同经验不要求相邻；不能因为列表里5在3前面，就把经验5的next_observation和经验3的observation拼成一条新的经验。旧代码只把完整Transition存入和取出，避免拆散字段。

## 连接机制

原生适配器把整数动作转换为Action，每次step请求1tick；collect_batch生成SamplingBatch；store_and_sample依次add，再按库存调用sample。缓冲区和随机对象都在批次循环外创建，跨两批复用。打印编号只用来追踪对象身份，不加入观察，也不是训练特征。

## 亲手实验

```bash
.venv/bin/python experiments/2026-09-10-chromium-replay-capacity/run.py --capacity 4
.venv/bin/python experiments/2026-09-10-chromium-replay-capacity/run.py --capacity 6
```

每次1个GUI、6个右移tick、0次训练，预计十几秒内结束。实际比较容量4/6的保留编号，并看再次抽样后内容是否仍在。教师两组接口检查已通过；学习者结果待报告，保持learning。

## 证据边界

教师原生两组均6条6tick，保存淘汰、抽样不删除和经验对应检查通过，未独立截图审阅画面。没有非零得分或真正终止样本，本次也不是缓冲容量对训练效果的评测。下一概念：把17项观察的抽样经验列表转换成批量张量。

2026-09-10课程重排：按用户要求，本项改为可选参考，不再是训练前置门槛。教师验证结果保留，未收到学习者两组结果，不记为学习者完成。下一项以[[教学计划]]为准，直接进入[[090-first-chromium-dqn-training]]，不再单开张量转换等重复基础课。
