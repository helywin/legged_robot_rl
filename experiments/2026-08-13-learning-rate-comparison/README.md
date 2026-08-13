---
title: 走格子环境的学习率对照实验
aliases:
  - 学习率对照实验记录
tags:
  - quadruped-rl/experiment
  - quadruped-rl/python
status: completed
created: 2026-08-13
updated: 2026-08-13
related:
  - "[[019-learning-rate-controlled-comparison]]"
  - "[[学习主页]]"
---

# 走格子环境的学习率对照实验

## 问题

在其他配置不变、训练轮数较少时，学习率怎样影响Q值吸收经验的速度和训练后策略？

## 假设

学习率为0时Q表完全不变；非零学习率可以学习；在当前确定性环境中，较大学习率会让Q值更快接近目标值。

## 环境与配置

- 证据级别：纯Python玩具环境受控实验
- Isaac Lab版本或提交：不适用，未启动Isaac Lab
- 任务名称：五格LineWorld
- 并行环境数：1
- 训练迭代数：每组20个episode
- 随机种子：0
- ε：0.20
- 折扣因子：0.90
- 评估：关闭探索、停止更新，每组10个episode
- 基线：学习率0.20
- 只改变的变量：学习率，取0、0.05、0.20、1.0

## 命令

```bash
python examples/train_line_world_q_learning.py --episodes 20 --learning-rate 0
python examples/train_line_world_q_learning.py --episodes 20 --learning-rate 0.05
python examples/train_line_world_q_learning.py --episodes 20 --learning-rate 0.2
python examples/train_line_world_q_learning.py --episodes 20 --learning-rate 1
```

## 产物

- 日志位置：本记录中的结果表；程序只输出到终端
- 检查点位置：无，Q表未保存到文件
- 图表或视频位置：无

## 观察与指标

| 学习率 | 训练成功 | 位置0-right | 位置3-right | 评估成功率 | 平均步数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4/20 | 0.000 | 0.000 | 0/10 | 10.00 |
| 0.05 | 18/20 | 0.005 | 0.603 | 10/10 | 4.00 |
| 0.20 | 18/20 | 0.372 | 0.982 | 10/10 | 4.00 |
| 1.0 | 14/20 | 0.702 | 1.000 | 10/10 | 4.00 |

学习率0时Q表全零，训练成功仅来自随机行为，关闭探索后失败。非零学习率均学到了每个位置向右。

## 结论

数据支持假设。在20轮限制下，较大学习率让关键Q值更快接近确定的目标值；学习率0完全不保存经验。

结论只适用于当前确定性五格环境、单个随机种子和本次训练长度。训练成功次数包含探索行为，不能视为学习率的直接刻度。

## 未验证与下一步

- 没有引入奖励噪声，未观察较大学习率可能造成的波动；
- 只有一个随机种子；
- 不能据此确定复杂任务的最佳学习率；
- 不涉及Gymnasium、Isaac Lab、四足机器人或真机。

下一步可只改变折扣因子，观察它如何改变远近位置的Q值；仍保持其他条件不变。
