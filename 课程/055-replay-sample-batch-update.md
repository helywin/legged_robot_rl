---
title: 把回放抽样接入一次批量DQN更新
aliases:
  - Replay sample batch update
  - DQN 回放抽样训练步
tags:
  - reinforcement-learning/python
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: learning
created: 2026-09-02
updated: 2026-09-02
related:
  - "[[054-replay-samples-to-tensors]]"
  - "[[053-pytorch-full-batch-dqn-update]]"
  - "[[概念/经验回放]]"
  - "[[概念/回放样本的字段批量化]]"
  - "[[概念/DQN训练流程]]"
  - "[[学习主页]]"
---

# 把回放抽样接入一次批量 DQN 更新

## 本课只接上三个已经学过的部件

前面三个部件单独运行时已经能工作：

1. `ReplayBuffer.sample(...)` 从旧经验中抽出一批 Python 对象；
2. `transitions_to_tensors(...)` 把对象列表整理成五个张量；
3. `full_batch_dqn_update(...)` 使用五个张量更新在线网络一次。

但“三个函数各自能跑”不等于“训练数据已经连通”。本课只回答：

> 上一个函数返回的数据，怎样成为下一个函数的输入？

这个过程不增加新公式，也不增加新的网络层。它只建立真实的数据流。

## 1. 先看每一段的输入和输出

### 第一段：从缓冲区抽样

本课缓冲区真实可运行的调用形式是：

```python
sampled_transitions = replay_buffer.sample(
    sample_size=batch_size,
    rng=random_generator,
)
```

`sample_size` 是要抽几条，`rng` 是用来产生随机选择的 `random.Random` 对象。返回值是：

```python
list[VectorTransition]
```

也就是“装着多条经验对象的 Python 列表”，还不是张量。

### 第二段：把经验列表变成张量

第 54 课已经完成的函数接口是：

```python
tensors = transitions_to_tensors(sampled_transitions)
```

它返回一个包含五项的 Python 元组：

```text
(
    observations,
    actions,
    rewards,
    next_observations,
    terminated,
)
```

顺序不是随意的：这就是函数在第 54 课承诺的返回顺序。

### 第三段：使用这五个张量更新

第 53 课已经完成的更新函数需要：

```python
full_batch_dqn_update(
    online_network,
    target_network,
    optimizer,
    observations,
    actions,
    rewards,
    next_observations,
    terminated,
    discount_factor,
)
```

前三项是训练对象，中间五项是刚刚组装出来的数据，最后一项是折扣因子。

## 2. Python 多个返回值怎样拆开

这里需要的 Python 语法叫“元组解包”。不要先记名字，先看一个最小例子：

```python
pair = (10, 20)
left, right = pair
```

第二行做的事等价于：

```python
left = pair[0]
right = pair[1]
```

所以执行后：

```text
left  = 10
right = 20
```

对本课的五个张量，展开写法是：

```python
tensors = transitions_to_tensors(sampled_transitions)

observations = tensors[0]
actions = tensors[1]
rewards = tensors[2]
next_observations = tensors[3]
terminated = tensors[4]
```

等价的解包写法是：

```python
observations, actions, rewards, next_observations, terminated = (
    transitions_to_tensors(sampled_transitions)
)
```

Python 会先执行右边的函数，得到五项元组，再从左到右一一交给五个变量。左边变量数量和右边项数不相等时，Python 会报错，不会自动猜测。

## 3. 用同一组数字跟完整个过程

缓冲区中保存三条经验：

| 缓冲区位置 | observation | action | reward | next observation | terminated |
| ---: | --- | ---: | ---: | --- | --- |
| 0 | `[0.2,-0.1]` | 1 | 0.2 | `[0.4,0.2]` | `False` |
| 1 | `[-0.3,0.4]` | 0 | 1.0 | `[-0.2,0.3]` | `True` |
| 2 | `[0.0,0.0]` | 0 | -0.2 | `[0.1,0.0]` | `False` |

固定使用 `random.Random(7)` 抽两条，抽样顺序是：

```text
先抽到缓冲区位置 1
再抽到缓冲区位置 0
```

因此 `sampled_transitions` 中的 observation 顺序是：

```text
[(-0.3, 0.4), (0.2, -0.1)]
```

组装后：

```text
observations      = [[-0.3,  0.4], [0.2, -0.1]]
actions           = [0, 1]
rewards           = [1.0, 0.2]
next_observations = [[-0.2, 0.3], [0.4, 0.2]]
terminated        = [True, False]
```

注意：抽样之后经验顺序变成了 `[1,0]`，但每条经验内部的观察、动作、奖励和结束标记仍然在同一行。随机化的是“经验行的顺序”，不是把各个字段各自打乱。

## 4. 这批数据进入网络后发生什么

固定参数的在线网络先得到：

```text
online_q_values = [
    [0.60,  0.70],   # 抽到的第 1 条经验
    [0.10, -0.05],   # 抽到的第 0 条经验
]
```

根据历史上实际执行的动作 `[0,1]` 逐行取值：

```text
selected_q_values = [0.60, -0.05]
```

目标仍然是第 53 课手算过的两个数，只是顺序跟抽样顺序一起变了：

```text
target_q_values = [1.00, 1.01]
```

逐条损失和平均损失：

```text
(0.60 - 1.00)²  = 0.1600
(-0.05 - 1.01)² = 1.1236

loss = (0.1600 + 1.1236) / 2 = 0.6418
```

虽然两条经验的顺序反了，平均损失没变。这是因为平均值只取决于这批里有哪些数，不取决于它们的先后顺序。但字段的行对齐仍然必须保持，否则会用别人的动作和奖励训练当前观察。

## 5. 哪些状态改变，哪些保持不变

| 对象 | 是否改变 | 原因 |
| --- | --- | --- |
| 回放缓冲区内容 | 不变 | `sample()` 只复制出一个抽样列表，不删除经验 |
| 抽样列表 | 新建 | 它记录本次选中的经验及顺序 |
| 五个批量张量 | 新建 | 它们由抽样对象的字段转换而来 |
| 在线网络参数 | 改变 | `backward()` 写入梯度，optimizer 读取后更新 |
| 目标网络参数 | 不变 | target 支路无梯度，且 optimizer 不管理它 |
| `random_generator` 内部状态 | 改变 | 生成一次抽样后，它会记住新的随机序列位置 |

这条因果链是：

```text
回放缓冲区
→ sample 选中两条完整经验
→ 字段组装产生五个张量
→ 张量进入在线支路和目标支路
→ loss 连着在线参数的计算图
→ backward 把梯度写入在线参数 .grad
→ optimizer.step 改变在线参数
```

`sample()` 和张量组装本身不会训练网络。真正改参数的仍然是最后两步 `backward()` 和 `optimizer.step()`。

## 6. 本课代码只负责调度

本课要补全的函数不重写抽样算法、张量组装算法或 DQN 公式。它只做三件事：

```text
调用已有抽样接口
→ 把返回值交给已有张量组装函数
→ 把五个张量交给已有批量更新函数
```

它的价值是保证接口顺序和数据责任没有接错，而不是把所有逻辑复制到一个大函数中。

## 本课训练

打开：

- `exercises/replay_sample_batch_update.py`

只修改 `train_from_replay()` 中的 TODO。练习文件已写明三个函数的完整接口、输入类型、返回顺序和调用目的。你需要亲手把三段连起来，不需要重写任何公式。

运行：

```bash
.venv/bin/python -m exercises.replay_sample_batch_update
```

成功时会确认：

- 固定随机种子确实抽到第 1、0 条经验；
- 抽样后的五个字段仍保持行对齐；
- 平均 loss 为 `0.6418`；
- 在线网络改变，目标网络和回放缓冲区不变。

## 当前边界

> [!success] 已有前置证据
> 学习者已完成第 54 课：五个张量的 shape、dtype、行对齐、单样本批次维和输入不变性全部通过。

> [!warning] 本课尚未完成
> 当前还没有学习者把随机抽样、字段组装和批量更新亲手连起来。也没有目标网络定期同步、完整 episode、CartPole 冒烟训练、检查点或独立评估。

## 一句话总结

一次回放训练步的核心是数据交接：抽样返回完整经验，组装函数返回五个张量，批量更新函数用它们建立 loss 到在线参数的计算图，最后只改变在线网络。

## 关联

- 前置数据组装：[[054-replay-samples-to-tensors|回放样本怎样变成批量张量]]
- 前置更新：[[053-pytorch-full-batch-dqn-update|完整批量 DQN 更新]]
- 总流程：[[概念/DQN训练流程]]
- 下一课：目标网络为什么只按间隔同步
