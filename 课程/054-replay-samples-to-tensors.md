---
title: 回放缓冲区样本怎样组装成批量张量
aliases:
  - Replay samples to tensors
  - DQN 回放批量组装
tags:
  - reinforcement-learning/python
  - reinforcement-learning/pytorch
  - reinforcement-learning/dqn
status: completed
created: 2026-09-02
updated: 2026-09-02
related:
  - "[[053-pytorch-full-batch-dqn-update]]"
  - "[[概念/回放样本的字段批量化]]"
  - "[[概念/经验回放]]"
  - "[[概念/批量张量与逐行动作索引]]"
  - "[[概念/终止掩码与批量DQN目标]]"
  - "[[概念/DQN训练流程]]"
  - "[[055-replay-sample-batch-update]]"
  - "[[学习主页]]"
---

# 回放缓冲区样本怎样组装成批量张量

## 本课只补上数据来源

第 53 课直接收到了五个已经整理好的批量张量：

```text
observations
executed_actions
rewards
next_observations
terminated
```

真实训练中，这些张量不是人工写出来的。环境每次只产生一条经验，经验回放保存许多条 Python 对象，再随机抽取其中几条。

本课只回答：

> 回放缓冲区抽出的 `Transition` 对象列表，怎样变成第 53 课能够直接使用的五个批量张量？

## 1. 一条 Transition 是一份完整记录

本课的一条经验对象可以想成一张有固定栏目的表单：

```python
VectorTransition(
    observation=(0.2, -0.1),
    action=1,
    reward=0.2,
    next_observation=(0.4, 0.2),
    terminated=False,
    truncated=False,
)
```

它保存一次环境交互的前因后果：

```text
旧观察和动作
→ 环境执行
→ 奖励、下一观察和结束信号
```

`Transition` 对象适合保存和阅读，但网络的线性层需要规则的数值张量，不能直接接收一组不同类型的 Python 属性。

## 2. 两条经验为什么不能直接变成一个大杂烩张量

固定抽到两条经验：

| 经验 | observation | action | reward | next observation | terminated |
| ---: | --- | ---: | ---: | --- | --- |
| 0 | `[0.2,-0.1]` | 1 | 0.2 | `[0.4,0.2]` | `False` |
| 1 | `[-0.3,0.4]` | 0 | 1.0 | `[-0.2,0.3]` | `True` |

这些字段形状和用途不同：

- observation 每条有两个浮点数；
- action 每条只有一个整数；
- reward 每条只有一个浮点数；
- terminated 每条只有一个布尔值。

如果把它们强行混进一个矩阵，动作编号和布尔标记可能被转成浮点数，观察边界也会丢失。正确做法是**同名字段纵向收集**。

## 3. 同名字段怎样纵向收集

先只看 observation。不要先记压缩写法，先用普通 `for` 循环把过程展开：

```python
observations_data = []

for transition in sampled_transitions:
    observations_data.append(transition.observation)
```

这四行的机制是：

1. `observations_data = []`：先建一个空列表，用来接收结果。
2. `for transition in sampled_transitions:`：从多条经验中每次取一条，当前这一条临时叫做 `transition`。
3. `transition.observation`：通过点号读取当前经验的 `observation` 属性。
4. `.append(...)`：把这个 observation 追加到结果列表末尾。

`transition` 不是 PyTorch 规定的名字，只是我们给“当前这条经验”取的临时变量名。`sampled_transitions` 才是原来的经验列表。一个用单数，一个用复数，是为了让两者容易区分。

用两条固定经验手动跟一遍：

| 循环次数 | `transition` 指向 | 取出的值 | `append` 之后的 `observations_data` |
| ---: | --- | --- | --- |
| 开始前 | 还没有 | 还没有 | `[]` |
| 1 | 第 0 条经验 | `(0.2, -0.1)` | `[(0.2, -0.1)]` |
| 2 | 第 1 条经验 | `(-0.3, 0.4)` | `[(0.2, -0.1), (-0.3, 0.4)]` |

所以这个循环没有计算 Q 值，也没有训练网络；它只是把每条经验里的同名属性收集起来。

### 列表推导式只是上面循环的短写

Python 3.6 已经支持这种写法：

```python
[
    transition.observation
    for transition in sampled_transitions
]
```

它的通用结构是：

```python
[每次要放进新列表的值 for 临时变量 in 原列表]
```

代入本题就是：

| 语法位置 | 本题代码 | 作用 |
| --- | --- | --- |
| 每次放入的值 | `transition.observation` | 取当前经验的观察 |
| 临时变量 | `transition` | 暂时代表当前经验 |
| 原列表 | `sampled_transitions` | 保存所有抽样经验 |

执行顺序仍然是：

```text
sampled_transitions 提供一条经验
→ 它暂时叫 transition
→ 读取 transition.observation
→ 把读到的值放进新列表
→ 继续下一条，直到遍历完
```

结果为：

```text
[
    [ 0.2, -0.1],
    [-0.3,  0.4],
]
```

两种写法生成的列表完全一样。如果你现在觉得列表推导式不直观，练习里可以继续使用普通 `for` 循环，不要硬背短写。

其他字段也采用同样原则，各自形成自己的列表，再转为张量。抽样顺序必须在所有字段中保持一致，否则第 0 行观察可能错误地配上第 1 行动作和奖励。

## 4. 先把 `torch.tensor` 接口完整拆开

这节练习不考 API 记忆。创建张量使用的完整形式是：

```python
tensor = torch.tensor(data, dtype=数据类型)
```

三个位置分别负责：

| 部分 | 含义 |
| --- | --- |
| `torch.tensor` | 调用 PyTorch 的张量创建函数 |
| `data` | 必须提供的 Python 数字、列表或嵌套列表 |
| `dtype=...` | 用名字明确指定张量中数字的类型 |

例如，把普通 Python 列表变成浮点张量：

```python
numbers = [0.2, 1.0]
rewards = torch.tensor(numbers, dtype=torch.float32)
```

可见结果是：

```text
rewards=tensor([0.2000, 1.0000])
rewards.shape=(2,)
rewards.dtype=torch.float32
```

`torch.tensor()` 不能空着调用，因为 PyTorch 不知道要把什么数据装进张量。

### 完整演示 observation 字段

练习文件已经用展开的普通 `for` 循环替你写好第一列：

```python
observations_data = []
for transition in transitions:
    observations_data.append(transition.observation)

observations = torch.tensor(
    observations_data,
    dtype=torch.float32,
)
```

第一步先得到普通 Python 数据：

```text
observations_data=[(0.2, -0.1), (-0.3, 0.4)]
```

第二步才由 `torch.tensor` 转换：

```text
observations=
tensor([[ 0.2000, -0.1000],
        [-0.3000,  0.4000]])
observations.shape=(2, 2)
observations.dtype=torch.float32
```

运行未完成的练习时，这组示范也会先打印出来。你需要照同一结构，把属性名和 dtype 换成另外四个字段。

> [!warning] 行末逗号不是装饰
> `x = 3` 得到整数 `3`，而 `x = 3,` 得到单元素元组 `(3,)`。写 `observations = torch.tensor(...),` 会让 `observations` 变成“装着一个张量的元组”，不再是张量本身。

## 5. 五个张量的 shape 为什么不同

| 张量 | 期望 shape | 每个维度含义 |
| --- | --- | --- |
| `observations` | `(2,2)` | 2 条经验，每条 2 个观察量 |
| `actions` | `(2,)` | 每条经验 1 个动作编号 |
| `rewards` | `(2,)` | 每条经验 1 个奖励 |
| `next_observations` | `(2,2)` | 2 条经验，每条 2 个下一观察量 |
| `terminated` | `(2,)` | 每条经验 1 个真正终止标记 |

第一维都表示同一批经验，行号必须一一对齐：

```text
第 i 行 observation
第 i 个 action
第 i 个 reward
第 i 行 next_observation
第 i 个 terminated
```

它们共同描述抽样列表中的第 $i$ 条经验。

## 6. dtype 不是随便选的

| 张量 | dtype | 原因 |
| --- | --- | --- |
| observations | `torch.float32` | 与线性层参数进行浮点计算 |
| actions | `torch.long` | `gather` 的列索引必须是整数索引类型 |
| rewards | `torch.float32` | 参与 target 数值计算 |
| next observations | `torch.float32` | 进入目标网络进行浮点计算 |
| terminated | `torch.bool` | 使用 `~` 取反，构造未来掩码 |

同一个数字 `1`，作为动作编号时是 `long`，作为奖励时是 `float32`，作为终止开关时是 `bool`。dtype 表示这个数字在计算中的职责，不只是显示形式。

## 7. truncated 为什么保存却不放进这五个 target 输入

回放经验仍应保存 `truncated`，因为它说明回合是否因时间上限停止。但第 53 课的 target 只需要 `terminated`：

- `terminated=True`：真正没有未来，未来掩码为 0；
- `truncated=True, terminated=False`：只是本回合停止，target 仍允许保留未来价值。

因此“环境循环是否 reset”和“target 是否去掉未来价值”使用不同判断。没有把 `truncated` 塞进终止张量，不代表忘记保存它。

## 8. 从环境到更新的完整位置

```text
环境 step 产生一条经验
→ Transition 保存字段
→ ReplayBuffer 保存许多 Transition
→ 随机 sample 得到对象列表
→ 按字段分别组装批量张量
→ 在线支路和目标支路完成批量更新
```

本课只负责倒数第二步。它不选择动作、不产生奖励、不计算 target，也不更新网络。

## 本课训练

打开：

- `exercises/replay_samples_to_tensors.py`

`observations` 已经作为完整示范写好。只修改 `transitions_to_tensors()` 中剩余的 TODO，照同一模式完成 actions、rewards、next observations 和 terminated。练习会检查所有字段的值、shape、dtype、单条批量边界和原经验对象不变性。

运行：

```bash
.venv/bin/python -m exercises.replay_samples_to_tensors
```

## 当前边界

> [!success] 已有前置证据
> 第 53 课已验证直接构造的五个批量张量能完成一次在线更新并保持目标网络不变。

> [!success] 学习者练习结果
> 学习者已把两条 `VectorTransition` 按同名字段组装为五个张量。实际运行确认 shape、dtype、行对齐、单条经验的批次维以及输入对象不变性全部通过。

> [!warning] 尚未验证
> 当前还没有把 ReplayBuffer 的随机抽样直接接到更新函数、运行 CartPole 完整训练、保存检查点或独立评估。

## 一句话总结

回放缓冲区抽出的是完整经验对象；训练前必须保持抽样顺序，按同名字段分别收集，并根据网络计算、动作索引和终止掩码的职责转换成正确 shape 与 dtype 的张量。

## 关联

- 前置：[[053-pytorch-full-batch-dqn-update|完整批量 DQN 更新]]
- 回放：[[概念/经验回放]]
- 概念：[[概念/回放样本的字段批量化]]
- 批量索引：[[概念/批量张量与逐行动作索引]]
- 下一课：[[055-replay-sample-batch-update|把 ReplayBuffer 抽样直接接入批量更新]]
