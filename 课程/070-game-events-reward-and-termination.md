---
title: 游戏事件奖励和回合结束怎样区分
status: completed
created: 2026-09-07
tags:
  - reinforcement-learning/environment
related:
  - "[[069-action-duration-and-observation-timing]]"
  - "[[概念/游戏事件与奖励终止]]"
---

# 游戏事件、奖励和回合结束怎样区分

本课只学：事件、训练分数和回合边界各管什么。采用假想多生命游戏，避免把未经核实的规则当作Chromium B.S.U.实测事实。

先读 [[概念/游戏事件与奖励终止]]。固定击毁2架、损失1条命、reward=-1，分别比较原有3条命和原有1条命。两者分数相同，却因是否仍有后续任务，让DQN target分别成为8和-1。

这节是已有reward、terminated与target知识在游戏任务定义中的原理澄清，不创建机械填空或固定答案检查器，也不算实际游戏训练。后续接真实接口时再提供可运行的实作。

下一概念：怎样验证游戏reset确实回到了可比较的起点。

2026-09-07：讲解后学习者要求继续，进入 [[071-reset-and-episode-boundaries]]。这里只记录概念课推进，不代表真实游戏接口或训练已通过。
