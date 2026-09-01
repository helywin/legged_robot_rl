---
title: 根据观察预测每个动作的 Q 值
aliases:
  - Q 值预测
  - 多动作 Q 值输出
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/dqn
  - reinforcement-learning/python
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[029-neural-network-parameters]]"
  - "[[031-q-prediction-error]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[学习主页]]"
  - "[[046-pytorch-two-action-q-values]]"
---

# 根据观察预测每个动作的 Q 值

## 本课目标

本课只理解“预测”：**使用当前参数，根据一个观察计算每个离散动作各自的 Q 值。**这里不是预测下一帧画面，而是估计现在选择各动作将来分别有多好。

## 最小接口

示例把观察简化成两个数字：

```text
observation = (位置, 危险程度)
```

动作顺序固定为：

```text
0 = LEFT
1 = RIGHT
```

因此：

```python
q_values = predict_q_values(observation, parameters)
```

返回：

```text
[LEFT 的 Q 值, RIGHT 的 Q 值]
```

这与 Q 表读取一行的后半段流程相同：取得一组 Q 值后，利用阶段可以选择最大值对应的动作，训练阶段仍可用 ε-greedy 探索其他动作。

## 实际运行

```bash
.venv/bin/python -m examples.shared_parameter_q_prediction
```

```text
observation=(0.2, 0.1) q_values=[0.1, 0.09] best=LEFT
observation=(0.8, 0.1) q_values=[-0.14, 0.51] best=RIGHT
observation=(0.8, 0.9) q_values=[0.5, 0.11] best=LEFT
parameters_changed=False
```

这次只改变观察，参数保持不变。不同观察产生了不同的 Q 值和最佳动作，预测结束后参数没有变化。

## 三个边界

1. 预测值只是当前估计，不保证正确；
2. 预测本身不会修改参数，因此还不是学习；
3. 输出最大 Q 值的动作是利用，ε-greedy 仍可能为了探索执行其他动作。

> [!success] 本课结论
> Q 表通过观察编号查出一行 Q 值；参数估计器通过观察现场计算一组 Q 值。二者都必须为每个可选动作提供一个值。

## 证据边界

本课只有标准库示例的启动检查。它没有证明预测正确，没有训练真实神经网络或 DQN。

## 关联

- 前置：[[029-neural-network-parameters|神经网络参数是共享的可学习数字]]
- 下一课：[[031-q-prediction-error|所选动作的 Q 值预测误差]]
- 概念：[[概念/神经网络参数与预测]]
- 上级：[[学习主页]]
- PyTorch 实现：[[046-pytorch-two-action-q-values|一组观察怎样输出两个动作 Q 值]]
