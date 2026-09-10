---
title: 采样结果怎样明确说明为什么停止
status: completed
created: 2026-09-09
tags:
  - reinforcement-learning/environment
related:
  - "[[085-termination-truncation-and-sampling-pause]]"
  - "[[084-consecutive-game-transitions]]"
  - "[[概念/游戏内部状态与策略接口]]"
---

# 采样结果怎样明确说明为什么停止

本课把085的边界写进返回值，复用084已有采样循环，不重新搭训练框架。当前先离线验证契约，后续原生游戏适配复用此接口。

## 为什么经验列表之外还要记录原因

一份列表有两条经验，只说明收到了两条，不能仅凭数量判断是采够了还是游戏结束。新返回对象SamplingBatch包含transitions列表和stop_reason字符串。它们分别回答“收到了什么”和“为什么停止”。这些元数据不加入网络观察，也不当奖励。

本课只可能返回两种原因：terminated表示本轮任务真正结束；batch_limit表示本批采够了，游戏仍可继续。当前不结束外部限定回合，所以不引入truncated，不能把batch_limit偷换成截断回合。

## 具体时间线

教学环境三步后结束，分数1000→1050→1050→1100。第一次预算2步，得到奖励[0.5,0]，最后terminated=False，原因batch_limit。再次对同一环境请求2步，只产生第三条奖励0.5且terminated=True，原因terminated；没有执行第四步。两批合起来仍是同一局，奖励总和1.0，前后观察相接。

换一个新环境，预算正好3步，第三步既用完预算又真的结束，仍应根据末条terminated记录真正结束。不能只写len(rows)==max_decisions就认定batch_limit。如果预算5步，也只收三条并报告terminated。

## 代码如何连接

collect_batch接受game、action_id和max_decisions。max_decisions是本批决策次数上限，不是回合时限，也不一定等于内部tick数；本题结果源每次step推进一个教学状态。

先建立重复动作tuple，交给已有collect_transitions得到rows，再查看rows[-1].terminated决定停止原因，最后返回SamplingBatch(transitions=rows,stop_reason=原因)。输入保证game可继续，预算验证为正整数，旧循环会保存终止步，故本次rows至少一条。

Python的(4,)*2得到(4,4)，这里乘法只是重复tuple元素，与函数签名单独的*不是同一语法。rows[-1]是列表最后一项。两项新返回字段用命名参数构造，旧经验对象不修改。

原数据产生链是现有采样器执行step并生成经验，新层读取真实末条标记，不能通过额外step探测结束，也不能修改末条标记来迎合自己的停止原因。报告terminated后，调用者不能再次对同一已结束环境采样。

## 亲手练习

文件：`exercises/sampling_stop_reason.py`，仅补全collect_batch。旧采样循环和返回数据类已导入。完整字段、调用方式、返回类型、手算时间线及边界在文件开头。

```bash
.venv/bin/python -m exercises.sampling_stop_reason
```

检查两批接续、恰好达到预算时真正结束、提前真正结束、单步小批次，以及动作和奖励未改变。学习者要实际复用采样结果生成新接口，不针对检查样本硬编码字符串。

## 证据与下一步

教师入口运行得到友好的未完成提示，无学习者新实现，保持learning。沿用084已验证的轨迹手算，不跑原生游戏或网络训练。下一概念：将原生游戏Action接口适配到这个采样器，分批采集真实经验。

2026-09-10：学习者实现collect_batch，实际运行全部离线检查通过，包括分批接续、预算恰好等于终止步数、提前终止与动作传递。使用batch[len(batch)-1]访问最后一条正确。086完成，进入[[087-real-game-sampling-batches]]。
