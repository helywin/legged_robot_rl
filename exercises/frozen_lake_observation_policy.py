#!/usr/bin/env python3
"""第 27 课编程练习：把固定动作序列改成观察策略。

为什么有这道题
============

上一题的动作序列可以在第一局走到终点，但它根据“这是第几次调用”选动作。
第二局重新 reset() 后，全局动作下标没有复位，因此会超出列表范围。

本题只改变一件事：让动作由当前 observation（角色所在格子的编号）决定。

场景
====

    观察编号                 地图

     0   1   2   3          S   F   F   F
     4   5   6   7          F   H   F   H
     8   9  10  11          F   F   F   H
    12  13  14  15          H   F   F   G

S 是起点，F 是安全冰面，H 是冰洞，G 是终点。地图没有随机打滑。

已有接口
========

choose_action(observation) 接收当前格子的编号，必须返回以下常量之一：

    LEFT、DOWN、RIGHT、UP

你的任务
========

只修改 choose_action() 中的 TODO。可以使用 if/elif，也可以使用字典。

要求：

1. 根据 observation 选择动作；
2. 不使用 global、动作下标或“第几次调用”；
3. 不修改 run_episode() 和 main()。

提示：先在地图上找出一条安全路线，再写出“观察编号 -> 动作”的对应关系。
策略不必为所有 16 个格子提供动作，只需覆盖你的安全路线会经过的格子。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.frozen_lake_observation_policy

成功条件
========

程序会在同一个 Python 进程中连续运行两局。两局都显示到达终点，最后显示：

    练习通过：观察策略在 reset() 后仍可重复使用

当前没有验证
============

本题仍是人工编写策略，不会学习或更新 Q 值。完成后只证明你理解了
“相同观察产生相同决策”，不代表 Q-learning、DQN 或游戏训练已经完成。
"""

from __future__ import annotations

from examples.frozen_lake import (
    ACTION_NAMES,
    ACTIONS,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FrozenLakeEnv,
)


def choose_action(observation: int) -> str:
    """仅根据当前观察返回动作；只修改这个函数。"""

    action_table = [
        (0, RIGHT),
        (1, RIGHT),
        (2, DOWN),
        (3, DOWN),
        (4, DOWN),
        (5, DOWN),
        (6, DOWN),
        (7, DOWN),
        (8, DOWN),
        (9, DOWN),
        (10, DOWN),
        (11, DOWN),
        (12, DOWN),
        (13, DOWN),
        (14, RIGHT),
        (15, DOWN),
    ]

    for act in action_table:
        if act[0] == observation:
            return act[1]


def run_episode(episode_number: int) -> bool | None:
    """运行一局；None 表示练习还没有填写。"""
    env = FrozenLakeEnv(max_steps=20)
    observation, _ = env.reset()
    print(f"第 {episode_number} 局开始")

    while not env.done:
        old_observation = observation
        try:
            action = choose_action(observation)
        except NotImplementedError as error:
            print(f"练习尚未完成：{error}")
            return None

        if action not in ACTIONS:
            print(f"策略返回了无效动作：{action!r}")
            return False

        observation, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={env.step_count:2d} "
            f"observation={old_observation:2d} "
            f"action={ACTION_NAMES[action]} "
            f"new_observation={observation:2d} "
            f"reward={reward:+.1f} "
            f"terminated={terminated} "
            f"truncated={truncated}"
        )

    if info["tile"] == "G":
        print(f"第 {episode_number} 局到达终点，共 {env.step_count} 步")
        return True

    if info["tile"] == "H":
        print(f"第 {episode_number} 局失败：在观察 {observation} 掉进冰洞")
    else:
        print(f"第 {episode_number} 局失败：20 步内没有到达终点")
    return False


def main() -> None:
    first_result = run_episode(episode_number=1)
    if first_result is None:
        return

    print()
    second_result = run_episode(episode_number=2)
    if first_result and second_result:
        print()
        print("练习通过：观察策略在 reset() 后仍可重复使用")


if __name__ == "__main__":
    main()
