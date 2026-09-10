---
title: 真实游戏怎样分批采集连续经验
status: completed
created: 2026-09-10
tags:
  - reinforcement-learning/experiment
related:
  - "[[086-sampling-stop-reason]]"
  - "[[概念/游戏内部状态与策略接口]]"
---

# 真实游戏怎样分批采集连续经验

086学习者实现已通过，本课把同一采样器接到原生游戏。实验见[[experiments/2026-09-10-chromium-sampling-batches/README]]；不创建新网络、观察字段或奖励因素。

## 先预测分组

固定连续6次RIGHT动作。每批2条时，分为动作1/2、3/4、5/6共3批；每批3条时，分为1/2/3、4/5/6共2批。每次原生step恰好1tick，因此两次都是6个动作、6tick。改变批次大小不会改变动作序列或重置世界。

关键检查是前一批末条next_observation等于后一批首条observation。每次调用collect_batch都传入同一个适配器对象，适配器持有同一个GameClient，因此没有在批间重开进程。batch_limit仅将经验列表交回调用者，不结束回合。

## 适配器做了什么

旧采样器使用整数动作编号；原生GameClient.step明确要求Action枚举。NativeGameAdapter.snapshot原样调用client.snapshot；step(action_id)先将整数转换为Action(action_id)，再调用client.step(...,ticks=1)，原样返回StepResult。Action(4)就是RIGHT，不是再选择一个动作。转换属于接口连接，不是网络推理，也不改变动作语义。

随后复用链为collect_batch→collect_transitions→适配器step→原生游戏；原生结果交回make_transition，观察与奖励仍由学习者完成的函数生成。每条经验包含当前实际动作及前后状态。整个调用链没有backward或optimizer，采到数据不等于训练了模型。

## 实际数字与逐项解释

教师两组均观测世界x约0→0.18→0.39→0.63→0.93→1.26→1.62。对应x观察为原值/10。分批2条时第一批末x约0.39，第二批下一步约0.63，不回到0；分批3条时第一批末x约0.63，第二批下一步约0.93。

两组分数没有增加，因此每条reward=(0-0)/100=0。这是有效的零奖励经验，不能为了看见正奖励而在本实验偷偷改变动作、添加开火或修改奖励。批间等待只影响观看速度，不增加tick；程序已检查等待快照相等。

## 亲手实验

每组1个GUI、0次训练、预计十几秒内结束，唯一改变每批条数：

```bash
.venv/bin/python experiments/2026-09-10-chromium-sampling-batches/run.py --batch-size 2
.venv/bin/python experiments/2026-09-10-chromium-sampling-batches/run.py --batch-size 3
```

不需要改代码。实际比较两组批数与总条数，报告是否为3批/2批且都6条，以及批间飞机是否接着移动。初始随机状态未固定，不要求所有原始字段跨运行相同。

## 证据与状态

教师已完成两组原生GUI模式接口运行，跨批衔接、停止原因、条数/tick数及奖励差值断言通过；未独立截图审阅画面。学习者结果待报告，learning。当前不覆盖原生真正终止、训练策略或完整游戏评测。下一概念：把真实采样结果接入经验回放保存与抽样。

2026-09-10：学习者回复“我测试了没问题”，确认两组分批采样实际对照符合要求。按学习者确认记为完成，不补写未提供的逐行日志或坐标。此结果证明当前短程采样连接可用，不代表网络训练、原生终止路径或完整游戏验收。
