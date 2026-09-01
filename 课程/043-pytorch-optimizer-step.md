---
title: 优化器怎样使用梯度更新参数
aliases:
  - PyTorch optimizer 入门
  - SGD 单步更新
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[042-pytorch-autograd-gradient]]"
  - "[[概念/优化器与参数更新]]"
  - "[[概念/自动求导与梯度]]"
  - "[[概念/预测误差与参数更新]]"
  - "[[学习主页]]"
  - "[[044-pytorch-zero-grad]]"
---

# 优化器怎样使用梯度更新参数

## 本课目标

长期方向仍是先完成小型 DQN，再走向 Chromium B.S.U.，并为后续 Go2 路线保留共同基础。本课只让优化器使用上一课得到的梯度更新一次参数；不编写训练循环，不处理经验回放，也不运行 DQN。

## 从“知道方向”到“真的移动”

上一课得到：

```text
weight.grad = -0.5
bias.grad = -1.0
```

这只是告诉我们参数应该往哪个方向变化。优化器像执行调整的人：它读取梯度，并按照学习率决定移动多少。

本课使用学习率 `0.1`。负梯度表示参数要增大，所以一次更新后：

```text
weight: 1.0 → 1.05
bias:   0.0 → 0.10
```

## optimizer 与 step

本课只增加两个 PyTorch 名称：

- `optimizer`：管理要更新的参数以及更新规则；
- `step()`：让优化器根据当前 `.grad` 真正修改一次参数。

创建最简单的 SGD 优化器：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

一次更新的核心顺序是：

```text
model(observation)
→ 计算 loss
→ loss.backward()
→ optimizer.step()
```

`backward()` 负责计算梯度，`step()` 才负责修改参数。这两个职责不能互相替代。

## 为什么损失会下降

更新前预测为 `0.5`，目标为 `1.0`，损失为 `0.25`。更新后：

```text
prediction = 0.5 × 1.05 + 0.10 = 0.625
loss = (0.625 - 1.0)² = 0.140625
```

预测从 `0.5` 向 `1.0` 靠近，损失从 `0.25` 降到约 `0.141`。这证明本次小步移动方向正确，但只证明这一条数据的一次更新。

运行教师示例：

```bash
.venv/bin/python -m examples.pytorch_optimizer_step_demo
```

## 本课训练

打开并只完成一个 `TODO`：

- `exercises/pytorch_optimizer_step.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_optimizer_step
```

练习要求先完成预测和损失，调用 `backward()` 计算梯度，再调用传入优化器的更新步骤。程序会检查参数确实改变、预测更接近目标且损失下降。

## 学习者练习结果

学习者按照“预测 → 损失 → `backward()` → `optimizer.step()`”完成了一次更新。实际运行结果：

```text
prediction_before=+0.500 expected=+0.500 PASS
loss_before=+0.250 expected=+0.250 PASS
weight_after=+1.050 expected=+1.050 PASS
bias_after=+0.100 expected=+0.100 PASS
prediction_after=+0.625 expected=+0.625 PASS
loss_decreased=True PASS

练习通过：优化器已使用梯度完成一次参数更新
```

这确认了学习者能够区分 `backward()` 的梯度计算职责和 `step()` 的参数修改职责，本课达到完成条件。

## 当前边界

> [!success] 启动检查
> 教师示例、学习者练习和自动测试确认 SGD 能使用一次梯度更新参数，并让当前样本的损失下降。

> [!warning] 尚未验证
> 还没有清空累计梯度、多轮训练循环、CartPole DQN 或独立评估。

## 一句话总结

`backward()` 计算梯度，优化器的 `step()` 按学习率使用梯度并真正修改参数。

## 关联

- 前置：[[042-pytorch-autograd-gradient|自动求导怎样计算参数梯度]]
- 概念：[[概念/优化器与参数更新]]
- 梯度：[[概念/自动求导与梯度]]
- 旧知识：[[概念/预测误差与参数更新]]
- 学习入口：[[学习主页]]
- 下一课：[[044-pytorch-zero-grad|为什么下一次更新前要清空旧梯度]]
