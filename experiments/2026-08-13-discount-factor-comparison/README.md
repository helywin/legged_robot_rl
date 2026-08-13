---
title: 走格子环境的折扣因子对照实验
aliases:
  - 折扣因子对照实验记录
tags:
  - quadruped-rl/experiment
  - quadruped-rl/python
status: completed
created: 2026-08-13
updated: 2026-08-13
related:
  - "[[020-discount-factor-controlled-comparison]]"
  - "[[学习主页]]"
---

# 走格子环境的折扣因子对照实验

## 问题

在其他配置不变时，折扣因子gamma怎样影响终点价值传播到较早位置，以及关闭探索后的策略。

## 假设

gamma为0时只看即时奖励，较早位置无法知道终点在右边；gamma越大，终点价值传到较早位置时保留得越多。

## 环境与配置

- 证据级别：纯Python玩具环境受控实验
- Isaac Lab版本或提交：不适用，未启动Isaac Lab
- 任务名称：五格LineWorld
- 并行环境数：1
- 训练迭代数：每组500个episode
- 随机种子：0
- ε：0.20
- 学习率：0.20
- 评估：关闭探索、停止更新，每组10个episode
- 基线：gamma=0.90
- 只改变的变量：gamma，取0、0.5、0.9、1.0

## 命令

```bash
python examples/train_line_world_q_learning.py --discount-factor 0
python examples/train_line_world_q_learning.py --discount-factor 0.5
python examples/train_line_world_q_learning.py --discount-factor 0.9
python examples/train_line_world_q_learning.py --discount-factor 1
```

## 产物

- 日志位置：本记录中的结果表；程序只输出到终端
- 检查点位置：无，Q表未保存到文件
- 图表或视频位置：无

## 观察与指标

| gamma | 训练成功 | 位置0-right | 位置1-right | 位置2-right | 位置3-right | 评估成功率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 255/500 | -0.010 | -0.010 | -0.010 | 1.000 | 0/10 |
| 0.5 | 497/500 | 0.107 | 0.235 | 0.490 | 1.000 | 10/10 |
| 0.9 | 497/500 | 0.702 | 0.791 | 0.890 | 1.000 | 10/10 |
| 1.0 | 497/500 | 0.970 | 0.980 | 0.990 | 1.000 | 10/10 |

gamma为0时，位置0到2的左右动作都只学到即时奖励约-0.01，评估无法选择通向终点的方向。gamma增加后，越远位置保留的终点价值越高。

## 结论

数据支持假设。gamma控制未来价值进入当前目标值的比例；它对远离终点的位置影响尤其明显。

结论只适用于当前确定性五格环境、500轮和随机种子0。训练成功包含探索行为，主要判断依据是Q表和关闭探索后的评估。

## 未验证与下一步

- 只有一个随机种子；
- 没有长时间、循环或奖励有噪声的任务；
- 不能据此确定复杂任务或机器狗训练的最佳gamma；
- 不涉及Gymnasium、Isaac Lab、四足机器人或真机。

完成本课并通过综合复习后，再判断纯Python Q-learning阶段是否达到教学计划中的过关标准。
