#!/usr/bin/env python3
"""第 26 课编程练习：为 FrozenLake 写一个手动策略。

本文件已经包含完成练习所需的全部信息，不需要来回查课程笔记。

场景
====

角色从 S 出发，只能在下面的确定性 4×4 地图中上下左右移动：

    观察编号                 地图

     0   1   2   3          S   F   F   F
     4   5   6   7          F   H   F   H
     8   9  10  11          F   F   F   H
    12  13  14  15          H   F   F   G

S 是起点，F 是安全冰面，H 是冰洞，G 是终点。
进入 H 会立即失败；进入 G 会立即成功；最多允许 20 步。

已有接口
========

环境已经实现，不需要修改：

    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(action)

策略只接收当前观察编号，并返回下面四个动作常量之一：

    LEFT、DOWN、RIGHT、UP

你的任务
========

只修改 choose_action() 函数中的 TODO，根据 observation 返回一个动作。
可以使用 if/elif，也可以使用字典。不要修改环境代码和 run_episode()。

这不是 Q-learning：本题先让你亲手写出一个策略，确认“策略在环境外根据观察选择动作”。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.frozen_lake_handwritten_policy

成功条件
========

终端最后显示“练习通过：到达终点”，并且过程中没有进入 H，也没有超过 20 步。

当前没有验证
============

完成本题只证明手写策略能通过这张固定地图，不代表 Q-learning 已经训练成功，也不代表
DQN、Chromium B.S.U.、Isaac Lab 或真实机器人已经得到验证。
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

index: int = 0

def choose_action(observation: int) -> str:
    """根据当前观察返回下一动作；只修改这个函数。"""
    del observation

    # TODO: 删除下一行，并写出你自己的策略。
    # raise NotImplementedError("请先补全 choose_action() 中的 TODO")
    global index
    action_table = [RIGHT,RIGHT,DOWN,DOWN,DOWN,RIGHT]
    ret = action_table[index]
    index += 1
    return ret


def run_episode() -> bool:
    """使用学习者写的策略运行一轮，并打印每一步。"""
    env = FrozenLakeEnv(max_steps=20)
    observation, info = env.reset()
    print("练习开始：请让 A 避开 H 到达 G")
    print(env.render())

    while not env.done:
        old_observation = observation
        try:
            action = choose_action(observation)
        except NotImplementedError as error:
            print()
            print(f"练习尚未完成：{error}")
            print("请打开本文件，阅读顶部场景并只修改 choose_action()。")
            return False

        if action not in ACTIONS:
            print(f"策略返回了无效动作：{action!r}")
            print("请返回 LEFT、DOWN、RIGHT、UP 之一。")
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
        print(env.render())

    if info["tile"] == "G":
        print(f"练习通过：到达终点，共 {env.step_count} 步")
        return True

    if info["tile"] == "H":
        print(f"练习未通过：在观察 {observation} 掉进冰洞")
    else:
        print("练习未通过：20 步内没有到达终点")
    return False


if __name__ == "__main__":
    run_episode()
