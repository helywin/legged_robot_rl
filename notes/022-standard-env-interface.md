---
title: 标准环境接口：reset 和 step 的约定
aliases:
  - 环境接口
  - 接口约定
tags:
  - quadruped-rl/python
  - quadruped-rl/learning
status: learning
created: 2026-08-27
updated: 2026-08-27
related:
  - "[[009-python-line-world]]"
  - "[[021-stage-one-q-learning-review]]"
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
python3 examples/env_interface_demo.py
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

## 关联

- 已有环境：[[009-python-line-world|纯 Python 走格子环境]]
- 阶段总结：[[021-stage-one-q-learning-review|纯Python Q-learning阶段综合复习]]
- 下一课（尚待创建）：Gymnasium——把这套约定变成行业标准的 Python 库
- 上级：[[学习主页]]
