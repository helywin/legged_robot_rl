---
title: 奖励怎样让策略更早注意小车偏移
status: completed
created: 2026-09-07
tags:
  - reinforcement-learning/reward
related:
  - "[[概念/居中奖励与策略取舍]]"
  - "[[066-first-real-cartpole-dqn-smoke-training]]"
---

# 奖励怎样让策略更早注意小车偏移

## 实验收尾（2026-09-07）

学习者完成奖励实现和真实实验，本课实作完成；效果不佳本身也是有效结果。

| 奖励 | 学习率 | 探索衰减步数 | 20回合平均原始步数 |
| --- | --- | --- | --- |
| 位置惩罚 | 0.001 | 20000 | 280.90 |
| 位置惩罚 | 0.0001 | 66666 | 377.75 |
| 原始奖励 | 0.0001 | 66666 | 401.55 |
| 原始奖励 | 0.0001 | 666666 | 185.65 |

各组均100000环境步、100000回放容量。同学习率和探索计划的对照中，原始奖励优于本次位置项；不能从同时改变两项配置的组间差异判断某一参数的独立作用。基准最终100回合验收未完成，按学习者要求调参告一段落。下面保留原始实验要求作为复习材料。

下一课：[[068-observation-needs-history|单帧画面为什么看不出运动方向]]。

本课只研究一个改动：给偏离中心的位置扣分。先阅读 [[概念/居中奖励与策略取舍]] 的完整手算与梯度链，再动手。

当前学习者日志：环境步和回放容量均为100000；20回合平均336.9步，没有满500步回合。学习者报告后期震荡和漂移。此前30000步的模型平均253.6，20回合均先出轨；不能直接把那个失败分类套到新模型，也不能把本次提升只归因于训练时长。

## 动手位置

`experiments/2026-09-07-cartpole-position-reward/train.py` 的 `position_reward` 是唯一 TODO。你需要亲手把位置惩罚的含义转为 Python；输入接口、算法含义、运行命令和成功标准都已写在文件里。接入回放和评估的基础设施已做好，不需要猜 Torch 接口。

```bash
.venv/bin/python experiments/2026-09-07-cartpole-position-reward/train.py baseline
.venv/bin/python experiments/2026-09-07-cartpole-position-reward/train.py shaped
```

两组都从同一随机种子重新训练100000环境步，每组约几十秒到数分钟。训练日志的return和冻结评估仍是原始存活步数；只有回放经验中的reward经过加工。每次自动保存到独立时间戳目录，保留旧产物。

终端会给出检查点位置，随后运行：

```bash
.venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/watch.py <本轮online-network.pt完整路径>
```

比较 `metrics.json` 的原始平均步数，以及 `reward-experiment.json` 的出轨/杆角失败次数、位置和角度的均方根。均方根是整段轨迹偏离零点的典型大小；平方后平均再开方，避免左右偏离互相抵消。它并不是振荡频率指标。

假设：位置项降低漂移且延长存活。如果更居中却更容易倒杆，说明出现了取舍，不应只看位置指标宣布成功。保留全部结果；在学习者完成实际实验前，本课保持learning。

实验记录：[[experiments/2026-09-07-cartpole-position-reward/README]]。
