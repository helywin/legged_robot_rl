---
title: 标准环境接口：reset 和 step 的约定
aliases:
  - 环境接口
  - 接口约定
tags:
  - quadruped-rl/python
  - quadruped-rl/learning
status: learned
created: 2026-08-27
updated: 2026-08-28
related:
  - "[[009-python-line-world]]"
  - "[[021-stage-one-q-learning-review]]"
  - "[[023-q-learning-for-games]]"
  - "[[025-choose-a-tabular-q-game]]"
  - "[[学习主页]]"
---

# 标准环境接口：reset 和 step 的约定

## 本课目标

理解"环境接口"这个概念：只要环境遵守同一套约定，一份训练循环代码可以不改一行地跑任何环境。这是之后使用 Gymnasium 和 Isaac Lab（Go2）的前提。

本课新词只有三个：

1. **接口（约定）**：环境和使用者之间"你提供哪些函数、返回什么形状数据"的约定；
2. **动作空间**：这个环境允许的全部动作清单；
3. **终止（terminated）与截断（truncated）**：两种不同的 episode 结束方式——真的到了终点 vs 只是步数上限到了。

## 类比：插座

电器不关心发电厂怎么发电，只要求插座是那个形状。训练循环就是电器，环境就是发电厂：`run_episode` 只要求环境有 `reset()` 和 `step()`，不关心里面是五个格子还是三个房间。

## 我们的约定长什么样

`examples/line_world.py` 其实已经基本遵守了这套约定：

| 约定项 | 我们环境里的样子 | 说明 |
| --- | --- | --- |
| `reset()` | 返回第一个观察（位置 0） | 每轮开始调用一次 |
| `step(动作)` | 返回 `(新观察, 奖励, done)` | 动作必须在动作空间里 |
| `env.done` 属性 | `True` 表示本轮结束 | 循环的退出条件 |
| 动作空间 | `{left, right}` | 环境拒绝清单外的动作 |

## 一个演示：同一份循环跑两个环境

`examples/env_interface_demo.py` 新写了 `RoomEnv`：观察是房间名字符串（`门口/客厅/卧室`），动作空间是 `{advance, retreat}`，奖励规则也不同。然后只用一个 `run_episode(env, policy)` 跑所有环境：

```python
def run_episode(env, policy):
    observation = env.reset()
    total_reward = 0.0
    steps = 0
    while not env.done:
        action = policy(observation)
        observation, reward, done = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward, steps
```

它没有读取任何环境的内部属性，所以观察是数字还是字符串、世界是几格，它都不在乎。

## 本次实际运行结果

```bash
.venv/bin/python -m examples.env_interface_demo
```

```text
环境1 LineWorldEnv：观察=数字位置，动作={left, right}
  总奖励=+0.97  步数=4  结束原因=到达目标
环境2 RoomEnv：观察=房间名字符串，动作={advance, retreat}
  总奖励=+0.98  步数=2  结束原因=到达目标
环境3 RoomEnv（一直后退，永远到不了卧室）
  总奖励=-0.12  步数=6  结束原因=步数用完

同一个 run_episode 跑了三个环境，一行都没改。
但结束原因必须为每个环境单独写判断，因为约定没有要求环境报告原因。
标准接口（Gymnasium）的做法：step() 直接返回 terminated 和 truncated。
```

## 约定的缺口：结束原因藏在环境肚子里

环境 3 的 episode 结束是"步数用完"，环境 1、2 是"到达目标"。但 `run_episode` 自己分不出来——它只能去偷看 `env.position` 或 `env.index`，而这两个环境的内部字段名不一样，判断代码无法复用。

标准接口（Gymnasium）因此把原因放进返回值里：

```text
我们的约定:  step(action) -> (观察, 奖励, done)
Gymnasium :  step(action) -> (观察, 奖励, terminated, truncated, info)
             reset()      -> (观察, info)
```

- `terminated=True`：到达终点，这一轮真的结束了；
- `truncated=True`：只是练到步数上限被叫停，机器人本身还有"未来"。

## 为什么 Q-learning 在乎这个区别

回忆 021 流程图里的公式 `y = r + γ·max Q(s',·)`：只有"真的没有下一步"时，未来那一项才应该按 0 算。

我们的代码用 `best_next_q = 0.0 if done else ...`，把截断也当成了没有未来。这是一个简化：被步数上限叫停时，其实还有未来价值，只是我们不再估计它。走格子环境里步数上限很少真正拦住学习，所以简化没事；但概念上要知道两种结束不一样。

## 往 Go2 的方向预告

Go2 的观察会变成几十个数字（关节角度、姿态等），动作空间变成 12 个关节的目标位置，但约定的骨架不变：`reset()` 给第一个观察，`step(动作)` 给新观察、奖励和结束标志。变的只是清单内容和长度。

> [!success] 当前证据
> 演示脚本实际运行成功，26 项单元测试全部通过（含本课新增 6 项）。这是纯 Python 层面的静态检查与启动检查，不涉及 Gymnasium、Isaac Lab、仿真机器人或真机。

## 思考题

1. `run_episode` 能不改一行跑三个环境，它依赖的是什么？如果某个环境的 `step()` 只返回 `(观察, 奖励)`，会发生什么？
2. "到达终点"和"步数用完"都让 `done=True`。对 Q-learning 更新公式里的 `γ·max Q(s',·)` 来说，这两种结束为什么不该同等对待？
3. 如果要给 `RoomEnv` 建一张 Q 表，应该是几行几列？"行"和"列"分别对应什么？观察是字符串而不是数字，有影响吗？
4. （开放题）Go2 的动作空间是 12 个关节目标，观察是几十个数字。接口约定里哪部分会变、哪部分永远不变？

## 学习者回答与纠正

### 第 1 题

学习者回答：三个环境依赖“共同的接口”。方向正确。进一步补全为：训练循环依赖环境共同提供 `reset()`、`step(action)`，并遵守约定的参数和返回值数量、顺序与含义。

对于 `observation, reward, done = env.step(action)`，学习者最初认为少返回 `done` 会让训练永远不能退出循环。当前代码的实际行为不是无限循环，而是 Python 在尝试把两个返回值拆给三个变量时立即报错：

```text
ValueError: not enough values to unpack (expected 3, got 2)
```

只有在代码已经改成接收两个返回值，同时环境的结束状态又永远不变时，才可能出现无法退出循环。这里要区分“接口形状不匹配导致立即报错”和“结束标志一直为假导致循环不停”。

### 第 2 题

学习者回答：未来价值表示离完成目标的分数；情况 A 已经到达终点，所以不再需要未来价值。

其中对情况 A 的结论正确：到达真正的终点后，这一轮没有后续动作，未来价值按 0 处理。需要纠正的是“未来价值”的定义：它不是“离目标还有多远的分数”，而是**从下一观察继续行动，预计还能获得的后续奖励**。距离目标更近有时会提高未来价值，但奖励也可能来自避开危险、收集物品或保持存活，因此两者不能画等号。

学习者随后补充：如果因为步数上限在位置 3 停止，把未来价值当成 0，会忽略“从位置 3 继续行动后预计还能获得的后续奖励”。回答正确。

本题结论：`terminated=True` 表示任务本身已经结束，没有后续动作；`truncated=True` 只表示本次练习被外部限制叫停，下一观察本身仍可能包含未来价值。两者都会结束当前 episode，但不能因此在 Q-learning 目标中不加区分地把未来都当成 0。

### 第 3 题

学习者回答：`RoomEnv` 的 Q 表是三行两列，行代表观察，列代表动作；观察使用字符串没有影响，因为它只是状态标志，并不用于实际计算。

表格形状和行列含义完全正确。最后一句的结论也正确，但理由需要更准确：观察不参与 Q 值的加减乘除，却会作为键或索引，用来定位 Q 表中的一行。使用字典时可以直接以 `"门口"` 等字符串为键；使用二维数组时则要先把三个字符串映射为 `0、1、2`。Q-learning 在意的是能否稳定区分观察，而不是观察本身必须写成数字。

### 第 4 题

学习者回答：换成 Go2 后，观察内容从环境位置变成环境姿态和机器人自身状态；`reset()` 仍负责重置环境，`step()` 仍按流程根据观察得到奖励、由策略选择动作并更新 Q 表。

第一部分方向正确，还需补上动作也发生了变化：从 `advance/retreat` 两个离散动作变成 12 个关节目标。

第二部分需要重新区分职责：

- `reset()` 属于环境，重置环境并返回初始观察；
- 策略在环境外，根据当前观察选择动作；
- `step(action)` 属于环境，只接收已经选好的动作，推进环境并返回新观察、奖励和结束信息；
- Q 表更新属于训练算法，在环境外根据这次交互结果执行。

因此，环境的 `step()` 不替策略选动作，也不更新 Q 表。

学习者最后把四项职责依次归类为“环境、策略、环境、训练算法”，全部正确，第 4 题通过。

## 本课结论

学习者已经能够：

1. 说明通用训练循环依赖共同接口；
2. 区分接口返回值错误与无法退出循环；
3. 区分真正终止和步数截断对未来价值的不同影响；
4. 根据观察数和动作数确定 Q 表形状；
5. 分清环境、策略和训练算法在闭环中的职责。

第 22 课完成。下一课进入 [[026-python-frozen-lake-environment|纯 Python FrozenLake 环境]]。

## 关联

- 已有环境：[[009-python-line-world|纯 Python 走格子环境]]
- 阶段总结：[[021-stage-one-q-learning-review|纯Python Q-learning阶段综合复习]]
- 游戏补充课：[[023-q-learning-for-games|Q-learning 能不能用来玩游戏]]
- 后续实践：[[025-choose-a-tabular-q-game|选择第一个表格 Q-learning 游戏]]之后的纯 Python FrozenLake
- 下一课：[[026-python-frozen-lake-environment|纯 Python FrozenLake 环境]]
- 上级：[[学习主页]]
