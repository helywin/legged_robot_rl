---
title: 一组观察怎样输出两个动作Q值
aliases:
  - PyTorch 多动作 Q 值
  - nn.Linear 两输入两输出
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[045-pytorch-training-loop]]"
  - "[[030-predict-action-q-values]]"
  - "[[概念/线性层与多动作Q值]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[学习主页]]"
  - "[[047-pytorch-selected-action-q-loss]]"
---

# 一组观察怎样输出两个动作 Q 值

## 本课目标

长期方向仍是先完成 CartPole 小型 DQN，再走向 Chromium B.S.U.，并为 Go2 路线保留共同基础。本课只把 PyTorch 模型从“一项观察、一个输出”扩展为“两项观察、两个动作 Q 值”；不增加隐藏层，不计算损失，也不训练环境。

## 一个动作一个分数

假设观察只有两个数字：

```text
observation = [位置, 变化趋势]
```

动作编号固定为：

```text
0 = LEFT
1 = RIGHT
```

模型应一次输出：

```text
q_values = [LEFT 的 Q 值, RIGHT 的 Q 值]
```

这与 Q 表读取一行后的结果相同，只是 Q 值现在由共享参数现场计算。

## nn.Linear 做什么

本课引入 `nn.Linear`：一个已经封装好权重和偏置的 PyTorch 线性层。

```python
nn.Linear(in_features=2, out_features=2)
```

- `in_features=2`：输入有两个观察量；
- `out_features=2`：输出有两个动作分数。

输入输出形状：

```text
observation.shape = (2,)
q_values.shape = (2,)
```

权重形状为 `(2, 2)`：第一行参数负责动作 0，第二行参数负责动作 1；偏置形状为 `(2,)`，每个动作各有一个偏置。

## 可见预测

教师示例使用固定参数。实际运行：

```bash
.venv/bin/python -m examples.pytorch_two_action_q_values
```

输出包含：

```text
weight_shape=(2, 2)
bias_shape=(2,)
observation=[0.2, -0.1] q_values=[0.1, -0.05] best_action=0
observation=[-1.0, 0.0] q_values=[-0.9, 1.2] best_action=1
```

不同观察可以改变两个 Q 值的大小关系，因此当前最佳动作也可能改变。

> [!warning] Q 值不是概率
> Q 值可以为负，也不要求两个值相加等于 1，所以这里不能添加 softmax。

## 本课训练

打开并完成 `TwoActionQExercise` 中的两个 `TODO`：

- `exercises/pytorch_two_action_q_output.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_two_action_q_output
```

你需要创建一个两输入、两输出的 `nn.Linear`，再在 `forward()` 中调用它。检查器会写入固定参数并验证三组 Q 值、最佳动作和预测不修改参数。

## 学习者练习结果

学习者创建了 `nn.Linear(2, 2)`，并在 `forward()` 中把观察交给该层。实际运行结果：

```text
linear_layer_registered=True PASS
parameter_shapes=(2, 2)/(2,) PASS
case1: q_values=[0.1, -0.05] best_action=0 PASS
case2: q_values=[-0.9, 1.2] best_action=1 PASS
case3: q_values=[0.1, 0.2] best_action=1 PASS
parameters_unchanged=True PASS

练习通过：两个观察量已映射为两个动作 Q 值
```

这确认了学习者能够用 PyTorch 线性层保持“输入数量、输出数量、参数形状和动作顺序”的对应关系，本课达到完成条件。

## 当前边界

> [!success] 启动检查
> 教师示例、学习者练习和自动测试确认一个线性层能把两个观察量映射成顺序固定的两个动作 Q 值。

> [!warning] 尚未验证
> 还没有只选择实际动作的 Q 值、损失更新、隐藏层、CartPole DQN 或独立评估。

## 一句话总结

`nn.Linear(2, 2)` 可以把两个观察量同时映射成两个动作 Q 值，输出位置必须与动作编号一致。

## 关联

- 前置：[[045-pytorch-training-loop|把一次参数更新放进多轮训练循环]]
- 旧知识：[[030-predict-action-q-values|根据观察预测每个动作的 Q 值]]
- 概念：[[概念/线性层与多动作Q值]]
- 参数与预测：[[概念/神经网络参数与预测]]
- 学习入口：[[学习主页]]
- 下一课：[[047-pytorch-selected-action-q-loss|只取实际动作的 Q 值计算损失]]
