---
title: 分数增量怎样成为一次动作的奖励
status: completed
created: 2026-09-09
tags:
  - reinforcement-learning/environment
related:
  - "[[082-real-bullet-step-observation]]"
  - "[[070-game-events-reward-and-termination]]"
  - "[[概念/游戏内部状态与策略接口]]"
---

# 分数增量怎样成为一次动作的奖励

082学习者确认预测与实际位置相同。真实快照与候选观察已接通，接下来补一条经验里的奖励连接。长期方向仍为游戏DQN，本课只采用分数增量演示，不把它当最终通关奖励，不启动网络训练。

## 为什么不直接拿累计分数

游戏player.score是累计分数。源码HeroAircraft::addScore按score+=in更新，newGame清零；补给也可加分，所以分数不是击中次数。如果某次动作把1000分变成1050分，这次新增50分。下一次仍为1050，没有新增，不能又把1050当成新奖励反复发放。

固定一组时间线1000→1050→1050→1100。用奖励尺度100，规定每新增100分对应1单位奖励：第一次(1050-1000)/100=0.5，第二次(1050-1050)/100=0，第三次(1100-1050)/100=0.5。总计1.0，正好等于(1100-1000)/100。先做差得到本次变化，再除尺度；不要拿动作后总分代替差值。

这是环境对该动作执行区间提供的反馈，并不证明加分只由这一瞬间的按键造成：前面发出的子弹也可能此刻击中。学习需要处理延迟反馈，不在此处强行识别因果事件。

## 与观察的职责分开

当前17项不含score，编码函数可以忽略它，而奖励函数仍读取同一快照的score。若两个快照只在分数上不同，observation与next_observation可以完全相同，reward仍可为0.5。观察回答策略看到什么，奖励回答本次区间获得什么反馈，两者不要求使用相同字段。

## 接成一条经验

make_transition接收动作前before、实际action_id、动作后的step_result。返回Transition对象的五个字段：

| 字段 | 来源 |
|---|---|
| observation | build_with_resources(before) |
| action | 传入的action_id |
| reward | 同一次动作后的score减之前score，再除reward_scale |
| next_observation | build_with_resources(step_result.snapshot) |
| terminated | step_result.terminated |

两个观察的编码、奖励相减只是读取数据，没有再次执行动作，也没有网络更新。Transition是给这五项命名的数据对象，可用Transition(observation=变量, action=变量, reward=变量, next_observation=变量, terminated=变量)构造；左边是字段名，右边是计算所得变量。核心连接由学习者完成，不预填完整函数。

## 边界

最后一步真正结束也应保留本次分数增量，不能把terminated=True时的reward清零。terminated来自实际结果，不根据零备用生命或分数自行猜测。前后必须来自同一游戏进程同一回合的这一动作，不能把重开后的0分减上一局的1100分；本题调用者保证这一边界，后续采样循环负责执行。

reward_scale=100是教学约定，区别于位置尺度10、控制尺度20。它只是奖励量级，不改变原游戏分数。本课没加生存或受伤奖励，因此尚未表达所有任务取舍，也没有宣称分数差足以训练成功。

## 手算对照与亲手练习

标准库算术已核对三次奖励[0.5,0.0,0.5]、总和1.0。练习为`exercises/game_step_transition.py`，仅补全make_transition；旧观察编码、样本和检查已经提供，函数没有星号参数。完整输入、构造方式、来源和成功条件写在文件开头。

```bash
.venv/bin/python -m exercises.game_step_transition
```

检查连续三个区间、只改分数时观察仍相同、动作后位置变化、奖励尺度转发和零备用生命不强制结束。题目实际生成可保存的经验对象，不是填写固定答案；没有写缓冲区或执行优化器。未完成入口能友好提示，学习者实现待提供，learning。

## 证据

静态核对score字段和游戏加分源码；标准库差值算术与脚手架入口运行通过。未改原生运行时、未设计完整奖励、未进行训练。下一概念：采样循环怎样把本轮next_observation接成下一轮observation。

2026-09-09学习者已完成make_transition，实际运行六个检查场景通过，包括分数差、最后一步奖励、前后观察、奖励尺度与零备用生命。学习者要求继续，083完成，进入[[084-consecutive-game-transitions]]。
