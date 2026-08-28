---
title: 纯 Python FrozenLake 环境与随机基线
aliases:
  - FrozenLake 环境
  - 第一次冰湖回放
tags:
  - game-rl/tabular-q
  - reinforcement-learning/environment
  - reinforcement-learning/python
status: learning
created: 2026-08-28
updated: 2026-08-28
related:
  - "[[022-standard-env-interface]]"
  - "[[025-choose-a-tabular-q-game]]"
  - "[[学习主页]]"
  - "[[教学计划]]"
---

# 纯 Python FrozenLake 环境与随机基线

## 本课目标

把上一课的标准环境接口落实成一个可以运行的 4×4 冰湖环境。本课只确认环境和随机策略，不训练 Q 表。

## 问题与唯一变化

- **问题**：标准 `reset/step` 接口能否表达带安全格、冰洞、终点和步数限制的二维小游戏？
- **唯一变化**：把一维走格子环境换成确定性的二维 FrozenLake；不加入打滑，不修改奖励，不进行训练。
- **预期现象**：随机策略可以完整运行，但可能撞墙、掉进冰洞或用完步数。
- **成功条件**：环境单元测试通过，固定种子回放能区分 `terminated` 和 `truncated`。

## Python 工程环境

项目由根目录 `pyproject.toml` 管理，使用仓库内 `.venv`：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --editable '.[dev]'
```

所有本课命令都使用 `.venv/bin/python`，不使用系统 Python。

## 地图与观察编号

```text
  0   1   2   3
  S   F   F   F

  4   5   6   7
  F   H   F   H

  8   9  10  11
  F   F   F   H

 12  13  14  15
  H   F   F   G
```

- `S`：起点；
- `F`：安全冰面；
- `H`：冰洞；
- `G`：终点；
- 观察是当前位置编号 `0～15`；
- 动作是 `left/down/right/up` 四种。

因此环境共有 16 个观察、4 个动作；以后使用的 Q 表会是 `16×4`。

## 标准接口

环境代码位于 `examples/frozen_lake.py`：

```text
reset()      → observation, info
step(action) → observation, reward, terminated, truncated, info
```

职责仍与第 22 课一致：

- 环境接收动作、推进位置并返回结果；
- 策略在环境外选择动作；
- 训练算法在环境外更新 Q 表。

## 终止与截断规则

- 到达 `G`：奖励 `+1`，`terminated=True`；
- 掉进 `H`：奖励 `0`，`terminated=True`；
- 停在安全格但走满 20 步：`truncated=True`；
- 本课是不打滑的确定性环境，同一个动作在同一格产生同一个移动结果。

## 实际运行

全量测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：35 项测试全部通过，其中 FrozenLake 新增 9 项，覆盖重置、边界、冰洞、终点、步数截断、终止优先、非法动作和渲染。

固定随机策略回放：

```bash
.venv/bin/python -m examples.frozen_lake --seed 0 --max-steps 20
```

结果：随机策略在第 10 步从观察 3 向下移动到观察 7，掉进冰洞：

```text
new_observation=7 reward=+0.0 terminated=True truncated=False
episode 结束: result=掉进冰洞 steps=10
```

> [!success] 当前证据
> 静态检查和纯 Python 启动检查通过；环境能区分到达终点、掉进冰洞和步数用完。当前只运行了随机策略，没有训练 Q 表，也没有验证 Gymnasium、DQN、Chromium B.S.U.、Isaac Lab 或真机。

## 动手编程

练习文件是 `exercises/frozen_lake_handwritten_policy.py`。场景、地图、已有接口、需要修改的位置、运行命令和成功条件全部写在该 Python 文件中，不需要从本笔记拼凑前提。

本次只补全一个手写策略，让角色根据当前观察选择动作并安全到达终点。它用于理解“策略在环境外选择动作”，不是 Q-learning 训练。

### 2026-08-28 完成记录

学习者写出的动作序列是：

```text
RIGHT → RIGHT → DOWN → DOWN → DOWN → RIGHT
```

实际回放路径为 `0 → 1 → 2 → 6 → 10 → 14 → 15`，第 6 步到达终点并获得 `+1` 奖励。按照本题已经写明的成功条件，本练习通过。修改后重新运行全量测试，35 项全部通过。

这份实现还揭示了一个值得单独学习的区别：代码删除了 `observation`，改用全局 `index` 依次取动作，因此它是依赖调用次数的**固定动作序列**。在同一 Python 进程中开始第二局时，`index` 没有复位，访问动作列表范围之外并触发 `IndexError`。

这不推翻本题的通过结果，但它给出了下一步的明确目标：把策略改为“当前位置相同，就给出相同动作”，即只根据 `observation` 查找动作，不依赖上一局留下的全局进度。做到这一点后，策略才能在每次 `reset()` 后重复使用。

> [!success] 本次证据
> 单局纯 Python 回放通过：6 步到达终点；仓库 35 项单元测试通过。

> [!warning] 证据边界
> 重复运行诊断未通过：第二局因全局动作下标未复位而报错。本次仍是手写策略，没有开始 Q-learning 训练。

## 关联

- 前置接口：[[022-standard-env-interface]]
- 选择原因：[[025-choose-a-tabular-q-game]]
- 上级：[[学习主页]]
- 路线：[[教学计划]]
