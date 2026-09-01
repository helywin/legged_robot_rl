---
title: 把一次参数更新放进多轮训练循环
aliases:
  - PyTorch 最小训练循环
  - 多步梯度更新
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[044-pytorch-zero-grad]]"
  - "[[概念/最小神经网络训练循环]]"
  - "[[概念/梯度累积与清零]]"
  - "[[概念/优化器与参数更新]]"
  - "[[学习主页]]"
  - "[[046-pytorch-two-action-q-values]]"
---

# 把一次参数更新放进多轮训练循环

## 本课目标

长期方向仍是先完成小型 DQN，再走向 Chromium B.S.U.，并为 Go2 路线保留共同基础。本课只把已经学会的一次更新放进 `for` 循环，观察同一个模型怎样连续八步逼近一个固定目标；不连接环境，也不运行 DQN。

## 为什么一次更新不够

一次更新把预测从 `0.5` 推到 `0.625`，方向正确，但距离目标 `1.0` 仍然很远。训练不是每次从头创建一个模型，而是让同一个模型保留刚更新的参数，再根据新的误差继续调整。

可以把它想成反复校准：每次测量还差多少，调整一小步，然后使用调整后的结果再次测量。

## 最小训练循环

每一步重复上一课已经学过的顺序：

```text
zero_grad()
→ prediction = model(observation)
→ loss
→ backward()
→ optimizer.step()
```

这里的“训练步”表示完成一次参数更新，不是环境的 `step()`。本课使用同一条固定数据重复八步。

教师示例的损失变化：

```text
step=01 loss_before_update=0.250000
step=02 loss_before_update=0.140625
step=03 loss_before_update=0.079102
...
step=08 loss_before_update=0.004454
final_prediction=0.949944
final_loss=0.002506
```

预测逐渐接近 `1.0`，损失持续下降，说明循环中每一步都沿用了前一步的新参数。

## loss 每轮都新建，怎样影响整个循环

每轮的 `prediction`、`loss` 和计算图都是临时对象；跨轮保留下来的是同一个模型及其参数。

```text
当前参数
→ 本轮 prediction
→ 本轮 loss
→ backward() 把梯度写入参数.grad
→ step() 修改参数
→ 下一轮用新参数重新产生 prediction 和 loss
```

因此，本轮 `loss` 不需要保存到下一轮。它的影响已经由 `optimizer.step()` 写进长期存在的模型参数。`losses.append(loss.item())` 只保存用于观察趋势的普通数字，不保存计算图。

## optimizer 为什么没有绑定 loss

优化器确实没有绑定 `loss`。创建它时：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

它保存的是模型参数对象的引用。前向计算使用同一批参数产生 `loss`，计算图记录这条依赖关系；随后 `loss.backward()` 把梯度写到参数的 `.grad`，`optimizer.step()` 再从同一批参数读取 `.grad`。

```text
loss ── backward() ──→ parameter.grad ←── step() ── optimizer
```

所以 `.grad` 是二者之间的桥：`loss` 不知道 optimizer，optimizer 也不需要知道 loss。

运行教师示例：

```bash
.venv/bin/python -m examples.pytorch_training_loop_demo
```

## 本课训练

打开并完成 `train_for_steps()` 中的一个 `TODO`：

- `exercises/pytorch_training_loop.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_training_loop
```

你需要亲手写出八步循环中的清梯度、预测、损失、反向计算、参数更新和损失记录。程序会检查记录数量、损失趋势和最终预测。

## 学习者练习与理解补充

学习者完成八步循环后，第一次用 `float(loss)` 记录损失，数值检查通过但 PyTorch 警告该张量仍连接自动求导图。改用 `loss.item()` 后，实际以警告视为错误重新运行仍全部通过：

```text
recorded_steps=8 PASS
loss_values_are_float PASS
first_loss=0.250000 PASS
loss_kept_decreasing=True PASS
final_prediction=0.949944 PASS
final_loss=0.002506 PASS

练习通过：单次更新已组成八步训练循环
```

学习者随后进一步追问“每轮 loss 都是新对象，`backward()` 怎样影响整个循环”以及“optimizer 在哪里绑定 loss”。这确认了需要区分三类关系：每轮临时计算图、跨轮保留的参数，以及作为二者桥梁的 `.grad`。相关机理已补充到本课和 [[概念/优化器与参数更新]]。

## 当前边界

> [!success] 启动检查
> 教师示例、学习者练习和自动测试确认固定数据上的八步训练能持续降低损失，并把预测推近目标。

> [!warning] 尚未验证
> 这里只拟合同一条数据，没有多观察输入、多个动作输出、CartPole DQN 或独立评估。

## 一句话总结

训练循环让同一个模型重复执行一次完整更新，每一步都从上一步学到的参数继续前进。

## 关联

- 前置：[[044-pytorch-zero-grad|为什么下一次更新前要清空旧梯度]]
- 概念：[[概念/最小神经网络训练循环]]
- 梯度清零：[[概念/梯度累积与清零]]
- 优化器：[[概念/优化器与参数更新]]
- 学习入口：[[学习主页]]
- 下一课：[[046-pytorch-two-action-q-values|一组观察怎样输出两个动作 Q 值]]
