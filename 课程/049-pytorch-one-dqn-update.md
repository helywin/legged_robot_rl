---
title: 把预测target和optimizer合成一次DQN更新
aliases:
  - PyTorch 单条经验 DQN 更新
  - 在线网络单步训练
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: completed
created: 2026-09-01
updated: 2026-09-02
related:
  - "[[047-pytorch-selected-action-q-loss]]"
  - "[[048-pytorch-dqn-target-q]]"
  - "[[概念/DQN完整训练流程与公式]]"
  - "[[概念/DQN训练流程]]"
  - "[[概念/优化器与参数更新]]"
  - "[[学习主页]]"
  - "[[050-pytorch-batch-selected-q]]"
---

# 把预测、target 和 optimizer 合成一次 DQN 更新

## 本课目标

长期方向仍是先完成 CartPole 小型 DQN，再走向 Chromium B.S.U.，并为 Go2 路线保留共同基础。本课不增加新算法部件，只把已经分别验证的在线预测、实际动作索引、目标网络 target、loss、backward 和 optimizer 拼成一条经验的一次更新。

这次先不背 API 顺序。我们只回答一个原理问题：

> 环境里已经发生的一次动作，怎样最终变成网络参数的一次变化？

## 1. DQN 到底在学习什么

Q 网络接收观察，并为每个动作输出一个 Q 值。Q 值不是“动作正确的概率”，而是网络对“从当前观察执行这个动作后，未来累计反馈有多好”的当前估计。

一条经验记录了实际发生的事情：

```text
(旧观察, 实际动作, 奖励, 下一观察, 是否真正终止)
```

这条经验不会直接告诉网络“正确参数是多少”。它只能先构造一个更可信的学习目标，再让旧预测向目标移动一点。因此一次更新的本质是：

```text
网络原来的估计 selected_q
        ↓ 与经验构造的 target_q 比较
误差
        ↓ 追溯哪些在线参数造成了这个误差
只把在线参数移动一小步
```

## 2. 为什么一条经验分成两条支路

一条经验包含：

```text
旧观察、实际动作、奖励、下一观察、terminated
```

在线支路产生需要训练的预测：

```text
旧观察 → 在线网络 → 全部 Q 值 → selected_q
```

目标支路产生不需要梯度的参照：

```text
下一观察 → 目标网络 → 最大下一 Q 值
奖励 + 折扣后的未来价值 → target_q
```

两条支路在损失处汇合：

```text
loss = (selected_q - target_q)²
```

在线支路回答“网络原来怎么估计这次实际动作”；目标支路回答“看过这次实际结果后，它现在应该靠近多少”。

这里不是拿奖励直接代替 Q 值。只要任务还没终止，下一观察之后仍有未来，所以目标是：

```text
target_q = 当前奖励 + 折扣因子 × 下一观察的最佳未来 Q 值
```

它把一个很长的未来问题拆成了“眼前奖励 + 从下一观察继续估计”。真正终止时已经没有下一步，目标才只剩当前奖励。

## 3. 用本课数字从头手算

固定经验是：

```text
observation      = [0.2, -0.1]
executed_action  = 1
reward           = 0.2
next_observation = [0.4, 0.2]
discount_factor  = 0.9
terminated       = False
```

在线网络先对旧观察输出：

```text
q_values = [0.10, -0.05]
```

动作 0 的值虽然更大，但环境当时实际执行的是动作 1。这条反馈评价的是动作 1，因此：

```text
selected_q = q_values[1] = -0.05
```

目标网络对下一观察输出：

```text
next_q_values = [0.90, -0.10]
best_next_q = 0.90
target_q = 0.2 + 0.9 × 0.90 = 1.01
```

这表示：在线网络原来认为动作 1 的未来价值是 `-0.05`，但这次经验给出的学习参照是 `1.01`。二者的平方损失为：

```text
loss = (-0.05 - 1.01)² = 1.1236
```

平方的作用不只是把负数变正数：偏差越大，损失增长越快；同时它的梯度会告诉我们预测应该往哪个方向移动。

## 4. backward 为什么能找到整个在线网络

前向计算不是只留下最终数字。PyTorch 同时记录了这个数字由哪些张量运算得到，这条依赖关系叫计算图：

```text
在线参数
   ↓ 参与旧观察的线性计算
q_values
   ↓ 按实际动作取索引 1
selected_q
   ↓ 与无梯度 target_q 计算平方差
loss
```

所以 `loss` 虽然刚刚创建，却已经通过 `selected_q`、`q_values` 连回了在线参数。循环本身不会被 `backward()` 修改；每一轮重新执行前向计算，就会为这一轮的新 `loss` 建立一张新的计算图。

`loss.backward()` 从 loss 反向沿图计算“每个在线参数改变一点，会让 loss 怎样变化”，并把结果累加到这些参数的 `.grad`。它只计算梯度，此时参数值还没有变化。

目标支路放在 `torch.no_grad()` 中，因此 `target_q` 是反向路径的终点：

```text
目标参数 → target_q    路径在这里断开，不写目标参数.grad
```

这就是目标网络不会被本次 `backward()` 顺手训练的原因。

## 5. optimizer 为什么不用绑定 loss

创建 optimizer 时传入的是在线网络的参数对象：

```python
optimizer = torch.optim.SGD(online_network.parameters(), lr=0.1)
```

它保存的是这些参数的引用，不保存 loss。两者通过同一批参数对象上的 `.grad` 接力：

```text
前向计算：在线参数 ─→ selected_q ─→ loss
反向计算：loss.backward() ─→ 在线参数.grad
参数更新：optimizer.step() 读取自己管理的参数.grad ─→ 改在线参数
```

所以 optimizer 不需要“知道 loss 是谁”。只要同时满足两件事，它就能工作：

1. loss 的计算图确实连到了这些在线参数；
2. optimizer 管理的正是这些参数对象。

如果 optimizer 绑定了另一个网络，即使当前 loss 的反向计算完全正确，它也不会更新本课的在线网络。

## 6. 为什么一次更新后是 0.1726

动作 1 的旧参数为：

```text
weight = [-1.0, 0.5]
bias = 0.2
```

它对旧观察的预测是：

```text
-1.0 × 0.2 + 0.5 × (-0.1) + 0.2 = -0.05
```

平方损失对预测值的梯度是：

```text
2 × (selected_q - target_q)
= 2 × (-0.05 - 1.01)
= -2.12
```

负梯度表示：如果把预测调大，损失会下降。这个梯度继续通过线性计算分给动作 1 的权重和偏置：

```text
weight.grad = -2.12 × [0.2, -0.1] = [-0.424, 0.212]
bias.grad = -2.12
```

SGD 使用学习率 `0.1`，按“旧参数减去学习率乘梯度”更新：

```text
new_weight = [-0.9576, 0.4788]
new_bias = 0.412
```

用新参数再次预测同一个旧观察：

```text
-0.9576 × 0.2 + 0.4788 × (-0.1) + 0.412 = 0.1726
```

`0.1726` 仍没有等于目标 `1.01`，因为学习率只允许走一小步；但它已经从 `-0.05` 朝正确方向移动。损失没有使用动作 0 的输出，所以这个无隐藏层示例中动作 0 对应的参数行保持不变。

## 7. 代码不是口诀，而是上述职责的对应物

| 代码                             | 背后的职责                     |
| ------------------------------ | ------------------------- |
| `optimizer.zero_grad()`        | 清掉上一次经验留在在线参数上的梯度         |
| `online_network(observation)`  | 产生当前预测，并建立 loss 回到在线参数的路径 |
| `q_values[executed_action]`    | 只训练环境当时真正执行的动作            |
| `calculate_dqn_target(...)`    | 用奖励和下一观察构造暂时固定的学习参照       |
| `(selected_q - target_q) ** 2` | 衡量当前预测离参照有多远              |
| `loss.backward()`              | 沿计算图把梯度写入在线参数的 `.grad`    |
| `optimizer.step()`             | 读取这些 `.grad`，实际改变在线参数     |

运行教师示例：

```bash
.venv/bin/python -m examples.pytorch_one_dqn_update
```

## 本课训练

如果你还没有看清这段代码在完整 DQN 中的位置，先阅读：[[概念/DQN完整训练流程与公式|DQN 完整训练流程、公式和 PlantUML 图]]。其中第二张图正好对应本练习的一个 `TODO`，但不会把完整答案提前写出来。

打开并完成 `one_dqn_update()` 中的一个 `TODO`：

- `exercises/pytorch_one_dqn_update.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_one_dqn_update
```

你需要把已经学过的组件按职责拼接。检查器会验证更新前预测、target、loss、更新后预测、未选在线输出行和目标网络。

## 学习者练习结果

学习者亲手补全了清梯度、在线预测、实际动作索引、无梯度 target、平方损失、反向计算、参数更新和返回值。实际运行结果：

```text
q_values_before=[0.1, -0.05] PASS
selected_q_before=-0.0500 PASS
target_q=1.0100 PASS
loss=1.1236 PASS
selected_q_after=0.1726 PASS
selected_q_moved_toward_target=True PASS
unselected_online_row_unchanged=True PASS
target_parameters_unchanged=True PASS
target_gradients_none=True PASS

练习通过：一条经验已完成一次 PyTorch DQN 在线更新
```

这次结果验证了完整接力关系：`loss.backward()` 把梯度写入在线参数的 `.grad`，只绑定在线参数的 optimizer 读取这些梯度并修改参数；目标网络没有梯度也没有参数变化。

## 当前边界

> [!success] 启动检查
> 教师示例、学习者练习和自动检查确认一条固定经验能更新在线网络的所选动作输出，并保持目标网络不变。

> [!warning] 尚未验证
> 还没有批量数据、隐藏层、经验回放抽样、目标网络定期同步、CartPole 训练、检查点或独立评估。

## 一句话总结

一条经验先用旧观察得到在线预测，再用奖励和下一观察构造固定参照；`backward()` 沿预测支路找出在线参数应怎样变化，optimizer 读取这些梯度后才真正更新参数。

## 关联

- 在线预测：[[047-pytorch-selected-action-q-loss|只取实际动作的 Q 值计算损失]]
- target：[[048-pytorch-dqn-target-q|奖励和下一观察怎样形成 DQN 目标值]]
- 流程：[[概念/DQN训练流程]]
- 完整图解：[[概念/DQN完整训练流程与公式]]
- 优化器：[[概念/优化器与参数更新]]
- 学习入口：[[学习主页]]
- 下一课：[[050-pytorch-batch-selected-q|一批经验怎样逐行取得实际动作 Q 值]]
