---
title: 在 FrozenLake 中训练 16×4 Q 表
aliases:
  - FrozenLake Q-learning
  - 二维 Q 表训练
tags:
  - game-rl/tabular-q
  - reinforcement-learning/q-learning
  - reinforcement-learning/python
status: completed
created: 2026-08-28
updated: 2026-08-28
related:
  - "[[027-observation-based-policy]]"
  - "[[016-full-line-world-q-learning]]"
  - "[[022-standard-env-interface]]"
  - "[[学习主页]]"
  - "[[教学计划]]"
---

# 在 FrozenLake 中训练 16×4 Q 表

## 本课目标

把已经在 LineWorld 使用过的 Q-learning 更新迁移到确定性 FrozenLake。长期目标仍是 DQN 游戏和四足机器人强化学习，但本课只验证表格算法能否通过相同的 `reset/step` 接口换一个环境。

## 问题与唯一变化

- **问题**：不再人工填写动作后，程序能否通过尝试和终点奖励学出一条安全路径？
- **唯一变化**：从手写“观察 → 动作”规则改为自动更新 `16×4` Q 表；地图、奖励和步数限制不变。
- **预期现象**：训练期间会因探索掉入冰洞，也会逐渐把终点价值传播到更早的观察；关闭探索后应稳定到达终点。
- **成功条件**：固定参数训练 1000 局后，停止更新并关闭探索，独立评估 20 局全部到达终点，且评估前后 Q 表不变。

## 为什么是 16×4

FrozenLake 有 16 个观察和 4 个动作，因此每个“观察—动作”组合都需要一个分数：

```text
Q 表行：0～15 共 16 个观察
Q 表列：LEFT、DOWN、RIGHT、UP 共 4 个动作
```

上一课的规则表只保存“这个观察最终选什么”。Q 表保存的是“这个观察下每个动作目前分别值多少”，训练策略再比较四个数。

## 本课编程题

打开 `exercises/train_frozen_lake_q_learning.py`。场景、函数输入、五个更新步骤、运行命令、成功输出和证据边界都已经写在文件顶部。

本次只补全 `learn_from_transition()`，其余训练和评估代码不修改。

运行：

```bash
.venv/bin/python -m exercises.train_frozen_lake_q_learning
```

> [!success] 当前状态
> `learn_from_transition()` 已完成，训练、冻结评估和 Q 表不变性检查均通过。

## 第一次失败与诊断

第一次实现正确算出了 `target`，也调用了 `update_q_value()`，但没有接住函数返回的新 Q 值，也没有写回 `q_table[observation][action_index]`。

实际现象是：

- 训练 1000 局只随机到达终点 10 次；
- 训练后 16×4 Q 表仍全部为 0；
- 冻结评估时四个动作并列，程序总是选择第一列 LEFT；
- 角色连续 20 步停在观察 0，评估结果为 `0/20`。

一条受控终止经验进一步确认了原因：观察 14 选择 RIGHT 到达观察 15、奖励为 1、学习率为 0.2 时，函数返回了目标值 1.0，但 Q 表中的值仍为 0；本次实际应写入的新 Q 值是 0.2。

## 修正后的实际结果

修正内容是接住 `new_q`、写回原 Q 表格子并返回新值。固定参数运行结果：

```text
训练期间到达终点: 525/1000
关闭探索、停止更新后的评估
到达终点: 20/20
平均步数: 6.00
首局路径: 0 -> 4 -> 8 -> 9 -> 13 -> 14 -> 15
评估是否修改 Q 表: False
练习通过：Q 表通过奖励更新学出了 FrozenLake 路线
```

受控终止经验也正确返回并写入 `0.2`。仓库全量单元测试为 35/35 通过。

> [!success] 本课结论
> Q-learning 更新不仅要计算目标和新 Q 值，还必须把新值写回正确的“观察—动作”格子。冻结评估关闭探索和更新后，学到的策略在确定性 FrozenLake 上稳定走出 6 步安全路线。

## terminated 与 truncated

两者都会让训练循环开始下一局，但计算未来价值时含义不同：

- `terminated=True` 表示进入 G 或 H，真正的任务状态结束，未来价值取 0；
- `truncated=True` 表示只因步数上限停止，当前格子不是终点，目标值仍可以参考新观察的 Q 值。

## 证据边界

> [!warning]
> 本课仍是纯 Python 表格 Q-learning。练习通过没有验证 Gymnasium、神经网络、DQN、Chromium B.S.U.、Isaac Lab 或真机。

## 关联

- 上一课：[[027-observation-based-policy]]
- 原训练循环：[[016-full-line-world-q-learning]]
- 环境接口：[[022-standard-env-interface]]
- 上级：[[学习主页]]
- 路线：[[教学计划]]
