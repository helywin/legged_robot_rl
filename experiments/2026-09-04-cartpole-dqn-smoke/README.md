---
title: 第一次真实CartPole DQN冒烟训练
aliases:
  - CartPole DQN smoke training
tags:
  - reinforcement-learning/experiment
  - reinforcement-learning/dqn
  - reinforcement-learning/cartpole
status: learning
created: 2026-09-04
updated: 2026-09-04
related:
  - "[[066-first-real-cartpole-dqn-smoke-training]]"
  - "[[概念/DQN小型基准]]"
  - "[[概念/训练与评估]]"
---

# 第一次真实 CartPole DQN 冒烟训练

## 问题

把此前分别验证的环境交互、经验回放、ReLU Q 网络、在线更新、目标同步和冻结评估接成真实闭环后，DQN 是否能让 `CartPole-v1` 的独立评估回报明显超过随机策略历史基线 `21.40`？

## 假设

如果数据和训练链已经正确接通，训练后的冻结策略应明显优于训练前的同一网络，也应超过随机基线。本次只是冒烟训练，预期超过平均回报 `200`，不要求立即达到最终验收线 `475`。

## 环境与配置

- 证据级别：真实 CartPole 冒烟训练与冻结评估
- Python：3.14.6，仓库 `.venv`
- PyTorch：2.13.0+cpu
- Gymnasium：1.3.0
- NumPy：2.5.2
- ONNX：1.22.0（学习者训练产物导出）
- ONNX Runtime：1.29.0（学习者产物数值对照）
- Isaac Lab 版本或提交：不适用
- 任务名称：`CartPole-v1`
- 并行环境数：1
- 环境步：30,000
- 在线更新：预填充达到 1,000 条后，每个环境步更新一次
- 网络：`4 → 64 → ReLU → 2`
- 回放容量：30,000
- batch size：64
- 折扣因子：0.99
- optimizer：Adam，学习率 0.001
- 目标网络同步：每 500 次在线更新复制一次
- 探索率：前 20,000 个环境步从 1.0 线性降到 0.05，之后保持 0.05
- 随机种子：20260904
- 冻结评估：训练前后使用相同的 20 个独立种子 `20270904`～`20270923`
- 基线：已记录随机策略 100 回合平均回报 21.40
- 只改变的变量：同一个网络是否经历上述 30,000 环境步训练

## 命令

教师参考实现：

```bash
.venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/train.py
```

学习者亲手补全训练器：

```bash
.venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/learner_train.py
```

学习者训练成功后，验证 ONNX 前向数值并打开 GUI 回放：

```bash
.venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/verify_onnx.py
.venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/watch.py
```

## 产物

- 脚本：`experiments/2026-09-04-cartpole-dqn-smoke/train.py`
- 学习者脚手架：`experiments/2026-09-04-cartpole-dqn-smoke/learner_train.py`
- GUI 回放：`experiments/2026-09-04-cartpole-dqn-smoke/watch.py`
- ONNX 对照：`experiments/2026-09-04-cartpole-dqn-smoke/verify_onnx.py`
- 日志位置：终端输出，以及 `artifacts/cartpole-dqn-smoke/metrics.json`
- 检查点位置：`artifacts/cartpole-dqn-smoke/online-network.pt`
- 学习者检查点：`artifacts/cartpole-dqn-from-scratch/online-network.pt`
- 学习者 ONNX：`artifacts/cartpole-dqn-from-scratch/online-network.onnx`
- 学习者指标：`artifacts/cartpole-dqn-from-scratch/metrics.json`
- 图表或视频位置：本次没有生成
- Git 边界：`artifacts/` 已被忽略，不提交模型和运行数据

## 观察与指标

教师参考运行结果：

| environment step | update count | episode count | epsilon | 最近20局平均回报 | 最近100次平均loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 4,001 | 212 | 0.762 | 35.00 | 0.3185 |
| 10,000 | 9,001 | 292 | 0.525 | 68.75 | 1.0290 |
| 15,000 | 14,001 | 316 | 0.288 | 232.80 | 2.5380 |
| 20,000 | 19,001 | 333 | 0.050 | 294.80 | 2.8385 |
| 25,000 | 24,001 | 350 | 0.050 | 298.25 | 3.7239 |
| 30,000 | 29,001 | 366 | 0.050 | 297.00 | 4.7614 |

冻结评估：

```text
训练前平均回报：9.20
训练后平均回报：253.60
随机策略历史基线：21.40
冒烟通过线：200.00 PASS
训练耗时：9.86 秒
```

`update_count=29001` 是因为从 environment step 1000 到 30000 均完成了一次更新，首尾都包含在内。

loss 上升而回报提高并不矛盾：训练后 target 的未来价值尺度变大，目标网络又会分段同步，每次随机批量的难度也不同。loss 衡量当前抽样预测与当前 target 的数值误差，不等于 episode return。

学习者运行结果：尚待记录。

## 结论

教师参考运行支持本次假设：同一网络训练前冻结评估平均回报为 `9.20`，训练后为 `253.60`，超过随机策略历史基线和本次冒烟通过线。这证明当前真实 DQN 数据与更新闭环能够产生明显学习效果。

证据只属于单随机种子、30,000 环境步的 CartPole 冒烟训练，以及 20 个固定种子的冻结评估。

## 未验证与下一步

本次教师参考没有达到最终预定的 100 回合平均回报至少 `475`、至少 90 回合达到 500 步；没有多训练种子复现、检查点选择或稳定性调参，也不验证 Isaac Lab 或真机。学习者实现、ONNX 数值对照和本机原生 GUI 回放仍待记录。

学习者亲自运行并读懂指标后，下一步才决定先解释 loss 与 return 的关系，还是只改变一个训练因素提高最终评估。
