---
title: CartPole 一步交互怎样变成回放经验
aliases:
  - CartPole step to transition
  - 环境交互生成经验
tags:
  - reinforcement-learning/python
  - reinforcement-learning/environment
  - reinforcement-learning/dqn
status: completed
created: 2026-09-03
updated: 2026-09-03
related:
  - "[[037-cartpole-interface-random-baseline]]"
  - "[[058-replay-warmup]]"
  - "[[概念/标准环境接口]]"
  - "[[概念/环境交互到经验记录]]"
  - "[[概念/经验回放]]"
  - "[[学习主页]]"
---

# CartPole 一步交互怎样变成回放经验

## 本课只解决一个断点

第 58 课已经能按照时间把 `transition_stream` 放进缓冲区，但那 6 条经验是事先写好的。真实 DQN 运行时没有别人替我们准备 `VectorTransition`，只有：

```text
reset() 给出的初始观察
step(action) 给出的动作后结果
```

本课只把这两部分拼成一条真实经验，不选择动作，也不训练网络。

## 1. 一条经验保存的是一个因果片段

先不用公式。想象我们要记录一次推杆实验：

```text
推之前杆是什么状态
→ 向左还是向右推
→ 推完得到什么奖励
→ 杆变成什么状态
→ 本局是否结束
```

如果只记推完后的状态，就不知道是什么动作造成的；如果只记推之前的状态和动作，就不知道后果。因此一条经验必须把同一次动作的前因和后果放在一起：

```text
VectorTransition(
    observation=动作前观察,
    action=实际执行动作,
    reward=动作后的奖励,
    next_observation=动作后观察,
    terminated=是否自然终止,
    truncated=是否被外部上限截断,
)
```

用下标写只是为了看清时间关系：

$$
T_t=(o_t,a_t,r_{t+1},o_{t+1},terminated,truncated)
$$

- $t$：执行本次动作之前的时刻；
- $o_t$：`reset()` 或上一轮 `step()` 留下的当前观察；
- $a_t$：策略在这个观察下实际执行的动作；
- $r_{t+1}$、$o_{t+1}$：环境执行动作后返回的奖励和新观察；
- 两个结束标记：说明能否继续调用 `step()`。

这里的 $t+1$ 不是数组下标操作，只表示这些结果是在动作之后产生的。

## 2. 用真实的第一步数字走完整链条

本课真实启动 `CartPole-v1`，固定：

```text
seed = 20260903
第一个 action = 1（向右推）
```

`reset()` 产生动作前观察：

```text
o0 = (-0.024423, -0.016661, -0.029921, 0.029134)
```

这四项依次是小车位置、小车速度、杆角度、杆角速度。执行 `action=1` 后，`step()` 实际返回：

```text
o1 = (-0.024756, 0.178877, -0.029338, -0.272838)
reward = 1.0
terminated = False
truncated = False
```

因此第 0 条经验必须是：

```text
observation      = o0
action           = 1
reward           = 1.0
next_observation = o1
terminated       = False
truncated        = False
```

它不是把六个互不相关的值塞进对象，而是在保存“看到 $o_0$ 后执行动作 1，环境变到 $o_1$”这个因果片段。

## 3. 为什么保存后必须移动当前观察

第一步结束后，环境已经处在 $o_1$，下一步不能继续把 $o_0$ 当作当前观察。正确顺序是：

```text
用 o0 和 step 结果构造第 0 条经验
→ 保存第 0 条经验
→ observation = next_observation
→ 当前 observation 现在是 o1
→ 再执行第 2 个动作
```

所以相邻经验应该首尾相接：

```text
第 0 条.next_observation == 第 1 条.observation
第 1 条.next_observation == 第 2 条.observation
```

如果把 `observation = next_observation` 提前到构造经验之前，第 0 条就会错误地变成：

```text
observation == next_observation == o1
```

这相当于记录“在动作后的状态执行了之前那个动作”，时间因果已经错位。代码仍可能运行，网络却会学习错误对应关系，所以检查器专门检查这条时间链。

## 4. 三步真实时间线

固定动作 `(1, 1, 0)` 后：

| 经验 | 动作前观察 | 动作 | 动作后观察 | 奖励 | 结束 |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | reset 得到的 $o_0$ | 1 | step 得到的 $o_1$ | 1.0 | 否 |
| 1 | 上一行的 $o_1$ | 1 | 新的 $o_2$ | 1.0 | 否 |
| 2 | 上一行的 $o_2$ | 0 | 新的 $o_3$ | 1.0 | 否 |

注意：动作序列不是学习结果，只是为了让数据可重现。它负责决定 `step()` 收到什么，CartPole 动力学负责计算动作后的新观察。

## 5. Gymnasium 原始接口与类型明确的包装

Gymnasium 原始调用是：

```python
observation, info = env.reset(seed=seed)

next_observation, reward, terminated, truncated, info = env.step(action)
```

由于通用 `gym.Env` 可以代表许多不同环境，IDE 对某些返回项只能显示宽泛类型。本课练习已经提供两个包装函数：

```python
observation = reset_cartpole(env=env, seed=seed)
```

VSCode 可以知道这里返回 `tuple[float, float, float, float]`。

```python
next_observation, reward, terminated, truncated = step_cartpole(
    env=env,
    action=action,
)
```

它的返回类型明确为：

```text
四项 float 观察、float 奖励、bool、bool
```

包装函数没有改变环境机理，只把 Gymnasium 数据转换成后续课程统一使用的 Python 类型。原始 `info` 没放进经验，是因为当前 DQN 更新公式不读取它；这不表示所有任务中的 `info` 都没用。

## 6. Python 的四项拆包在做什么

下面一行左边有四个名字：

```python
next_observation, reward, terminated, truncated = step_cartpole(...)
```

等价于先接住整个返回元组，再按位置取值：

```python
step_result = step_cartpole(...)
next_observation = step_result[0]
reward = step_result[1]
terminated = step_result[2]
truncated = step_result[3]
```

拆包不会复制环境，也不会执行四次 `step()`；右边函数只调用一次，然后四项结果分别绑定给四个名字。

## 7. 为什么观察要转换成 float 元组

Gymnasium 的 CartPole 观察原本是 NumPy 数组。练习中的包装函数使用：

```python
tuple(float(value) for value in observation)
```

普通循环写法是：

```python
converted_values = []
for value in observation:
    converted_values.append(float(value))
converted_observation = tuple(converted_values)
```

两者都逐项把数值变成普通 Python `float`，再组成不可变元组。这样缓冲区保存的是本步数值快照，不依赖之后对 NumPy 数组对象的处理。

旧课的二维观察和本课的四维观察长度不同，因此通用 `VectorTransition` 的注解扩展为：

```python
observation: tuple[float, ...]
```

这里的 `...` 表示“可以有任意数量的 float”，不是元组中真的存了一个省略号。CartPole 包装函数仍用四项精确类型，防止少一项或多一项。

## 8. 代码与机制逐项对应

| 代码 | 读取什么 | 产生或保存什么 | 会不会训练网络 |
| --- | --- | --- | --- |
| `reset_cartpole(...)` | 种子和环境 | 第一项当前观察 | 不会 |
| `step_cartpole(...)` | 当前环境状态和动作 | 新观察、奖励、结束标记 | 不会 |
| `VectorTransition(...)` | 本步前因与后果 | 一条经验对象 | 不会 |
| `replay_buffer.add(...)` | 经验对象 | 缓冲区多一条记录 | 不会 |
| `observation = next_observation` | 本步新观察 | 下一步的当前观察名字 | 不会 |
| `break` | 两个结束标记 | 停止本局动作循环 | 不会 |

这一课完全没有 `loss`、`backward()` 或 `optimizer.step()`，所以网络参数不会改变。它只负责给后续训练提供来源正确的数据。

## 本课训练

打开：

- `exercises/cartpole_step_to_transition.py`

只修改 `collect_cartpole_transitions()` 中的 TODO。文件已经给出：

- 两个返回类型明确的 CartPole 接口；
- `VectorTransition` 的完整构造接口；
- 三步固定数据及因果顺序；
- 终止后立即停止的边界；
- 全部成功条件。

运行：

```bash
.venv/bin/python -m exercises.cartpole_step_to_transition
```

你需要亲手完成并运行通过，本课才会标记完成。

## 当前边界

> [!success] 学习者练习结果
> 学习者已把真实 `reset/step` 返回值组成 `VectorTransition`。实际运行确认三条经验、动作和奖励顺序、第一步真实数值、相邻观察首尾链、缓冲区快照以及终止后立即停止 `step()` 全部通过。

> [!warning] 尚未验证
> 固定动作只用于采集数据，不是探索策略；本课也没有让 CartPole 四项观察进入网络、执行抽样更新、完成 episode、保存检查点或独立评估。

## 一句话总结

一条回放经验必须在执行动作前保留当前观察，再把同一次 `step()` 产生的奖励、新观察和结束标记接在后面；保存完成后，才能把新观察变成下一步的当前观察。

## 关联

- 前置：[[058-replay-warmup|环境为什么要先填充回放缓冲区再训练]]
- 环境接口：[[概念/标准环境接口]]
- 经验结构：[[概念/环境交互到经验记录]]
- 回放缓冲区：[[概念/经验回放]]
- 下一课：[[060-cartpole-four-input-linear-q-network|CartPole 四项观察怎样变成两个动作 Q 值]]
