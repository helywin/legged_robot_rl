---
title: 为什么下一次更新前要清空旧梯度
aliases:
  - PyTorch zero_grad 入门
  - 梯度累积
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[043-pytorch-optimizer-step]]"
  - "[[概念/梯度累积与清零]]"
  - "[[概念/自动求导与梯度]]"
  - "[[概念/优化器与参数更新]]"
  - "[[学习主页]]"
  - "[[045-pytorch-training-loop]]"
---

# 为什么下一次更新前要清空旧梯度

## 本课目标

长期方向仍是先完成小型 DQN，再走向 Chromium B.S.U.，并为 Go2 路线保留共同基础。本课只观察第二次 `backward()` 怎样与旧梯度相加，以及怎样在新数据前清空旧梯度；不编写完整训练循环。

## 旧调整意见为什么会干扰新数据

可以把梯度想成一张“这次应该怎样调参数”的纸条。第一次更新结束后，PyTorch 默认不会扔掉纸条。第二次调用 `backward()` 时，新意见会直接加到旧意见上。

这叫**梯度累积**。它有专门用途，但如果我们希望第二条数据独立决定下一次更新，忘记清空就会得到错误的合计方向。

## 两个方向相反的目标

第一次目标是 `1.0`，留下权重梯度 `-0.5`，要求增大参数。完成第一次更新后，预测变为 `0.625`。

第二次目标改为 `0.4`。因为 `0.625` 高于目标，这一次的新权重梯度应为正数 `+0.225`，要求减小参数。

如果不清空：

```text
旧梯度 -0.500 + 新梯度 +0.225 = 实际 grad -0.275
```

旧方向压过新方向，第二次预测反而从 `0.625` 升到约 `0.694`，离新目标更远。

如果先调用：

```python
optimizer.zero_grad()
```

第二次只保留新梯度 `+0.225`，预测会降到约 `0.569`，向目标 `0.4` 靠近。

## 正确顺序

对每一条希望独立更新的数据，顺序是：

```text
zero_grad()
→ model(observation)
→ 计算 loss
→ backward()
→ step()
```

`zero_grad()` 清理的是旧梯度，不会把学到的参数恢复成初始值。

运行教师对照示例：

```bash
.venv/bin/python -m examples.pytorch_zero_grad_demo
```

## 本课训练

打开并只完成一个 `TODO`：

- `exercises/pytorch_zero_grad_before_update.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_zero_grad_before_update
```

后面的预测、损失、反向计算和参数更新已经写好。你只需要在正确位置清空第一次留下的梯度。程序会检查第二次梯度方向、更新后预测以及是否向第二个目标靠近。

## 学习者练习结果

学习者把 `optimizer.zero_grad()` 放在第二次预测与 `backward()` 之前。实际运行结果：

```text
prediction_before=+0.625 expected=+0.625 PASS
loss_before=+0.051 expected=+0.051 PASS
weight_gradient=+0.225 expected=+0.225 PASS
bias_gradient=+0.450 expected=+0.450 PASS
prediction_after=+0.569 expected=+0.569 PASS
moved_toward_target=True PASS

练习通过：第二次 backward 前已清空旧梯度
```

这确认了第二次更新只使用当前目标产生的梯度，而不是把第一次梯度意外累加进来；本课达到完成条件。

## 当前边界

> [!success] 启动检查
> 教师对照示例、学习者练习和自动测试确认：不清空时第二次梯度会混入旧值；清空后第二次更新会向当前目标移动。

> [!warning] 尚未验证
> 还没有多轮训练循环、批量训练、CartPole DQN 或独立评估。

## 一句话总结

`backward()` 默认累加梯度；希望新数据独立决定更新时，要在本轮计算前用 `zero_grad()` 清空旧梯度。

## 关联

- 前置：[[043-pytorch-optimizer-step|优化器怎样使用梯度更新参数]]
- 概念：[[概念/梯度累积与清零]]
- 梯度：[[概念/自动求导与梯度]]
- 优化器：[[概念/优化器与参数更新]]
- 学习入口：[[学习主页]]
- 下一课：[[045-pytorch-training-loop|把一次参数更新放进多轮训练循环]]
