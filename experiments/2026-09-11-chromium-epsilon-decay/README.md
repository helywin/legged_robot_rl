---
title: 按全局决策数衰减探索率
status: learning
tags: [强化学习, Chromium, 探索]
---

## 问题与假设

用户同意将预填后固定0.2改为线性衰减：前期探索更多，后期降低随机动作干扰。
本轮仅改训练探索规则及显示，不改36维观察、奖励、gamma、学习率或评测贪心策略。
不把短训练接线通过解释为策略改善。

## 第一版规则与边界（已被下方预算比例规则替代）

默认epsilon_start=1.0，epsilon_end=0.05，epsilon_decay_decisions=100000。
第1..256次动作（预填阶段）epsilon为1；之后用动作前已完成的决策数计算：
elapsed=max(0,decisions_before_action-256)。
未满100000时epsilon=1-0.95×elapsed/100000，达到后直接返回0.05。

动作前累计完成256、50256、100256次决策，epsilon分别1、0.525、0.05。
多环境第i个动作使用decisions+i（从0计数），不是每轮并行调用只计一次。
开新局、目标网络同步、经验池环回不重启衰减。每个新训练进程仍从头训练，不支持续训。
40000次更新约对应40255决策，末次epsilon约0.620019，不会自动加速衰减到终点。

设置epsilon_start=epsilon_end=0.2可恢复预填后固定0.2的历史性能对照；
两个既有benchmark脚本已显式固定，以免其对照规则随默认值改变。

## 环境与配置

CPU真实无窗口8环境，短训练2000更新。实际配置：

```json
{
  "max_updates": 2000,
  "update_backend": "eager",
  "num_envs": 8,
  "task_version": "task-v6-36",
  "episode_limit": 1000,
  "capacity": 10000,
  "batch_size": 32,
  "learning_starts": 256,
  "target_sync_every": 100,
  "epsilon_start": 1.0,
  "epsilon_end": 0.05,
  "epsilon_decay_decisions": 100000,
  "gamma": 0.99,
  "learning_rate": 0.001,
  "game_seed": 31,
  "network_seed": 7,
  "replay_seed": 7,
  "exploration_seed": 11
}
```

## 完整命令

```bash
.venv/bin/python exercises/chromium_dqn/train.py --check
.venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -p 'test_epsilon_schedule.py' -v
.venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -p 'test_parallel_training.py' -v
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 2000 --num-envs 8
```

## 产物与结果

- 训练目录：/home/jiang/code/legged_robot_rl/exercises/chromium_dqn/runs/train-96489d791642475599094051fc659f44，含config.json、steps.jsonl、summary.json、policy.pt。
- 实际2255决策、2000更新，末次epsilon=0.981019。
- 全部2255条日志的epsilon按其decision序号重新计算，逐项一致；真实权重加载通过。
- 新增2项测试通过：默认边界/非法配置，以及单环境与4环境在多次reset和经验池环回后曲线相同。
- 编程检查器train --check和2项并行经验池测试通过。
- 短训练未跑满10万决策，衰减终点由自动测试验证，不声称已完成长训练。
- 进度条保留近期平均奖励，并显示最近一次已记录动作使用的epsilon。

## 结论与下一步

调度与日志接线通过，没有证明已收敛或躲避改善；未做GUI回放。
后续长训练结合中间冻结评测判断，而不是只看含随机探索的训练均奖。


## 按用户纠正改为总预算比例

固定100000方案取消，当前epsilon_decay_fraction=0.8。
总决策数learning_starts-1+max_updates；预填后剩余max_updates-1条，
衰减跨度max(1,ceil(剩余×0.8))，约最后20%决策保持0.05。
只有预填的极小预算保持全随机。不新增需要同时修改的停止参数。

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 1000 --num-envs 8
.venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -p 'test_epsilon_schedule.py' -v
.venv/bin/python exercises/chromium_dqn/train.py --check
```

3项探索率测试与循环检查通过。真实1000更新、1255决策短训练完成，
终端显示最终ε=0.0500，日志暂存/tmp/chromium-epsilon-budget.log。
其runs/train-9acff93527374543a507a72a0fee9cbb目录在后续读取时已不在工作区，
因此本次仅按保留的终端日志确认运行完成，不声称逐条复核该次经验日志。
本轮不代表策略收敛。
