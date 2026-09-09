---
title: 本轮结束状态怎样接成下一轮开始状态
status: completed
created: 2026-09-09
tags:
  - reinforcement-learning/environment
related:
  - "[[083-score-delta-transition]]"
  - "[[概念/游戏内部状态与策略接口]]"
---

# 本轮结束状态怎样接成下一轮开始状态

083学习者实现已通过，接下来只把单条经验放入连续采样循环。不添加网络、奖励因素或新动作策略。

## 当前问题

如果连续调用三次step，不能每次都用最初的snapshot作为before。每条经验必须描述该动作紧邻的前后状态。本轮step_result.snapshot就是下一轮的before，这是本课唯一核心连接。

## 同一条数字时间线

四份教学快照S0、S1、S2、S3的累计分数分别为1000、1050、1050、1100；飞机x分别0、1、2、3。动作依次4、0、3，第三个动作结束回合。这条教学轨迹是预先准备的结果源，不模拟动作的物理效果，不能解释为IDLE导致飞机右移或LEFT产生正位移。

| 本轮 | before | after | 分数增量/100 |
|---|---|---|---|
| 1 | S0：1000 | S1：1050 | 0.5 |
| 2 | S1：1050 | S2：1050 | 0.0 |
| 3 | S2：1050 | S3：1100 | 0.5 |

每轮结束后让before指向after，所以第三轮只计1050到1100之间的50分。若忘记更新，一直使用S0，则奖励错误地成为0.5、0.5、1.0，总计2.0，重复奖励以前已获得的分数。正确总计1.0。两组标准库算术已运行核对。

观察也要接力：第1条next_observation等于第2条observation，第2条next_observation等于第3条observation；对应x观察为0→0.1、0.1→0.2、0.2→0.3。不是每条都从0开始。

## Python里的状态接力

先用game.snapshot()取得一次当前状态，局部变量叫before。每轮调用game.step(action_id)得到本轮结果，把before、action_id和结果交给已完成的make_transition，再append到经验列表。随后用before = result.snapshot接上新状态。这个赋值只改变局部变量引用，不修改原快照或已经保存的经验。

真正结束的那一步仍要先保存，再停止；先判断结束就直接跳出会丢掉最后一步的奖励。如果动作列表为(4,0,3,9)，前三步中第三步已结束，最后的9不应执行。若只提供(4,0)，列表耗尽时仍未真正结束，不能擅自把最后一条terminated改成True。

## 亲手练习

文件：`exercises/consecutive_game_transitions.py`。只补全collect_transitions，返回Transition组成的列表。ScriptedGame已提供snapshot/step两个方法，是有明确末端的离线教学结果源；make_transition复用083实现。题目包含完整接口、处理顺序、预期输出、允许修改位置和运行方法，不使用星号参数。

```bash
.venv/bin/python -m exercises.consecutive_game_transitions
```

成功条件是增量不重复、相邻观察衔接、终止步保留、终止后不执行剩余动作、动作预算耗尽不伪造终止，以及再次采集从当前状态继续。采样列表不是网络更新，尚未写入经验回放或执行优化器。

## 证据与下一步

教师完成标准库算术和未完成脚手架入口检查，学习者新实现待提供，保持learning。当前为离线采样器连接，不宣称真实游戏训练或奖励方案已验证。下一概念：真正终止与人为采样上限怎样分别记录。


## 学习者完成记录与等价实现

2026-09-09实际运行全部检查通过。学习者选择每轮先before=game.snapshot()再step，然后构造经验、append、检查terminated并break。这与通过赋值接力的写法在当前不自行推进的结果源/同步游戏中等价，且输入预算耗尽后再次调用也从当前状态继续。接受实现，不强制重写为单次snapshot。对于持续实时推进的环境，额外读取可能跨越未记录的更新，不能无条件推广；当前接口无此自动推进。

已同步放宽题目表述，明确两种实现的适用条件。084完成，进入[[085-termination-truncation-and-sampling-pause]]的原理澄清，无网络训练。
