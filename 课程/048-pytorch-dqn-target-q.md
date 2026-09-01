---
title: 奖励和下一观察怎样形成DQN目标值
aliases:
  - PyTorch DQN target
  - 目标网络最大未来 Q 值
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[047-pytorch-selected-action-q-loss]]"
  - "[[034-target-network]]"
  - "[[概念/DQN目标Q值]]"
  - "[[概念/目标网络]]"
  - "[[概念/标准环境接口]]"
  - "[[学习主页]]"
  - "[[049-pytorch-one-dqn-update]]"
---

# 奖励和下一观察怎样形成 DQN 目标值

## 本课目标

长期方向仍是先完成 CartPole 小型 DQN，再走向 Chromium B.S.U.，并为 Go2 路线保留共同基础。本课只把奖励、下一观察和 `terminated` 组成 PyTorch target；不连接在线网络的损失，也不更新参数。

## target 回答什么问题

上一课的 `selected_q` 是在线网络目前的预测。target 则回答：根据这条经验刚得到的奖励和下一观察，这个预测现在应该靠近多少？

固定数据：

```text
reward = 0.2
discount_factor = 0.9
next_q_values = [0.9, -0.1]
```

任务没有真正结束时，下一观察仍有未来：

```text
best_next_q = 0.9
target = 0.2 + 0.9 × 0.9 = 1.01
```

`terminated=True` 时没有未来动作：

```text
target = reward = 0.2
```

## 为什么使用目标网络

下一观察的 Q 值由暂时固定的目标网络计算。在线网络负责当前 `selected_q`，目标网络负责提供未来估计，避免当前预测和目标同时被同一次 optimizer 更新推动。

```text
旧观察 → 在线网络 → selected_q ─┐
                                  ├→ loss
下一观察 → 目标网络 → target ────┘
```

## 为什么 target 不记录梯度

target 是本次在线网络要追赶的参照，不是这次要优化的参数来源。因此计算时使用：

```python
with torch.no_grad():
    ...
```

最终 `target.requires_grad` 应为 `False`，目标网络参数的 `.grad` 应保持 `None`。

## terminated 与 truncated

只有 `terminated=True` 才把未来价值设为零。如果只是 `truncated=True`，当前 episode 虽然因外部上限停止，但下一观察仍可能具有未来价值，计算 target 时仍走非终止分支。

运行教师示例：

```bash
.venv/bin/python -m examples.pytorch_dqn_target_q
```

## 本课训练

打开并完成 `calculate_target()` 中的一个 `TODO`：

- `exercises/pytorch_dqn_target_q.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_dqn_target_q
```

你需要在无梯度环境中处理终止与非终止两条路径。检查器会验证两个 target 数值、梯度隔离和目标参数不变。

## 学习者练习结果

学习者实现了终止与非终止两条路径；非终止时在 `torch.no_grad()` 中调用目标网络并选取最大下一 Q 值。实际运行结果：

```text
next_q_values=[0.9, -0.1] PASS
continuing_target=1.010 PASS
terminal_target=0.200 PASS
target_requires_grad=False PASS
target_network_gradients_none=True PASS
target_parameters_unchanged=True PASS

练习通过：奖励与下一观察已组成无梯度的 DQN 目标值
```

当前一维示例中使用 Python `max(q_values)` 也能返回正确标量张量；进入批量数据后需要改用按维度的张量 `max`。本课的单条经验目标计算已经达到完成条件。

## 当前边界

> [!success] 启动检查
> 教师示例、学习者练习和自动测试确认非终止 target 为 `1.01`、终止 target 为 `0.20`，且目标网络不产生梯度。

> [!warning] 尚未验证
> 还没有把 target 与在线 `selected_q` 组成损失和 optimizer 更新，也没有经验回放抽样、CartPole 训练或独立评估。

## 一句话总结

非终止 target 使用奖励加折扣后的最大下一 Q 值，真正终止 target 只使用奖励，并且 target 一侧不参与反向传播。

## 关联

- 前置：[[047-pytorch-selected-action-q-loss|只取实际动作的 Q 值计算损失]]
- 目标网络：[[034-target-network|为什么 DQN 需要目标网络]]、[[概念/目标网络]]
- 概念：[[概念/DQN目标Q值]]
- 环境结束：[[概念/标准环境接口]]
- 学习入口：[[学习主页]]
- 下一课：[[049-pytorch-one-dqn-update|把预测、target 和 optimizer 合成一次 DQN 更新]]
