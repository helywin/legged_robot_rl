#!/usr/bin/env python3
"""第 28 课编程练习：让 FrozenLake 的 Q 表自动更新。

本文件包含完成练习所需的全部信息。你只需要修改一个函数，不需要自己搭环境、
训练循环或评估循环。

本课只改变一件事
================

上一课的“观察 -> 动作”对应关系由人填写。本课仍使用相同的确定性 4×4 地图，
但先创建全零的 16×4 Q 表，再让每一步经验自动更新其中一个 Q 值。

地图与 Q 表
===========

    观察编号                 地图

     0   1   2   3          S   F   F   F
     4   5   6   7          F   H   F   H
     8   9  10  11          F   F   F   H
    12  13  14  15          H   F   F   G

Q 表有 16 行、4 列：

    q_table[观察编号][动作下标]

动作列的固定顺序是：

    0=LEFT、1=DOWN、2=RIGHT、3=UP

例如 q_table[6][1] 保存“处于观察 6 时选择 DOWN”的 Q 值。

已有训练流程
============

程序已经实现下面的循环：

1. env.reset() 返回 observation；
2. epsilon-greedy 根据 q_table[observation] 选择 action_index；
3. env.step(action) 返回 new_observation、reward、terminated、truncated；
4. 调用 learn_from_transition() 更新刚才那个 Q 值；
5. observation = new_observation，继续下一步。

程序还会在训练后关闭探索、停止更新，独立评估 20 局并打印一局路径。

你的任务
========

只修改 learn_from_transition() 中的 TODO。不要修改训练参数、环境、训练循环或评估。

函数的输入：

- q_table：需要被更新的 Q 表；
- observation：执行动作前的观察；
- action_index：刚才选择的动作列；
- reward：环境这一步返回的奖励；
- new_observation：执行动作后的观察；
- terminated：是否到达真正的终点 G 或冰洞 H；
- learning_rate：学习率；
- discount_factor：折扣因子。

按下面五步编写代码：

1. 读出旧值 q_table[observation][action_index]；
2. 如果 terminated 为 True，best_next_q 是 0；否则取新观察那一行的最大 Q 值；
3. 调用 calculate_target() 计算目标值；
4. 调用 update_q_value() 计算新 Q 值；
5. 把新值写回原来的 Q 表格子，并返回这个新值。

这里没有要求使用 `del` 删除任何输入。

为什么函数只看 terminated
==========================

训练循环遇到 terminated 或 truncated 都会结束当前一局，但两者含义不同：

- terminated：已经进入 G 或 H，这个任务状态真的结束，后面没有未来价值；
- truncated：只是 20 步上限到了，当前格子本身不是终点，计算目标时仍允许参考
  new_observation 的 Q 值。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.train_frozen_lake_q_learning

未完成时程序会显示 TODO 提示，不会输出异常堆栈。

成功条件
========

正确实现后，固定参数会训练 1000 局，并关闭探索评估 20 局。终端最后应同时显示：

    到达终点: 20/20
    练习通过：Q 表通过奖励更新学出了 FrozenLake 路线

训练期间成功次数不要求等于 1000，因为 epsilon-greedy 会继续探索。

当前没有验证
============

完成本题只验证纯 Python、确定性 FrozenLake 上的表格 Q-learning。它不代表已经使用
Gymnasium、神经网络、DQN、Chromium B.S.U.、Isaac Lab 或真实机器人。
"""

from __future__ import annotations

import random

from examples.frozen_lake import ACTIONS, ACTION_NAMES, FrozenLakeEnv
from examples.q_learning_single_update import calculate_target, update_q_value
from examples.train_line_world_q_learning import (
    choose_action_index,
    create_q_table,
)


QTable = list[list[float]]


def learn_from_transition(
    q_table: QTable,
    observation: int,
    action_index: int,
    reward: float,
    new_observation: int,
    terminated: bool,
    learning_rate: float,
    discount_factor: float,
) -> float:
    """用一条经验更新一个 Q 值；只修改本函数中的 TODO。"""

    old_q = q_table[observation][action_index]
    best_next_q: int
    if terminated:
        best_next_q = 0
    else:
        best_next_q = max(q_table[new_observation])
    target = calculate_target(reward, best_next_q, discount_factor, terminated)
    new_q = update_q_value(old_q, target, learning_rate)
    q_table[observation][action_index] = new_q
    return new_q

def train_q_table(
    episodes: int = 1000,
    epsilon: float = 0.3,
    learning_rate: float = 0.2,
    discount_factor: float = 0.95,
    seed: int = 0,
) -> tuple[QTable, int]:
    """训练多局，返回 Q 表和训练期间到达终点的次数。"""
    env = FrozenLakeEnv(max_steps=20)
    q_table = create_q_table(env.observation_count, len(ACTIONS))
    rng = random.Random(seed)
    successes = 0

    for _ in range(episodes):
        observation, _ = env.reset()

        while not env.done:
            action_index = choose_action_index(
                q_table[observation], epsilon, rng
            )
            action = ACTIONS[action_index]
            (
                new_observation,
                reward,
                terminated,
                _truncated,
                info,
            ) = env.step(action)

            learn_from_transition(
                q_table=q_table,
                observation=observation,
                action_index=action_index,
                reward=reward,
                new_observation=new_observation,
                terminated=terminated,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            observation = new_observation

        if info["tile"] == "G":
            successes += 1

    return q_table, successes


def evaluate_q_table(
    q_table: QTable, episodes: int = 20
) -> tuple[int, float, list[int]]:
    """关闭探索且不更新 Q 表，返回成功数、平均步数和首局路径。"""
    successes = 0
    total_steps = 0
    first_path: list[int] = []

    for episode_index in range(episodes):
        env = FrozenLakeEnv(max_steps=20)
        observation, _ = env.reset()
        path = [observation]

        while not env.done:
            action_index = max(
                range(len(ACTIONS)), key=q_table[observation].__getitem__
            )
            observation, _, _, _, info = env.step(ACTIONS[action_index])
            path.append(observation)

        total_steps += env.step_count
        if info["tile"] == "G":
            successes += 1
        if episode_index == 0:
            first_path = path

    return successes, total_steps / episodes, first_path


def print_q_table(q_table: QTable) -> None:
    """打印 16×4 Q 表和每行的最佳动作。"""
    print("\n训练后的 Q 表")
    print("观察 |   left |   down |  right |     up | 最佳动作")
    print("-----+--------+--------+--------+--------+---------")
    for observation, q_row in enumerate(q_table):
        best_index = max(range(len(ACTIONS)), key=q_row.__getitem__)
        print(
            f"{observation:4d} | "
            f"{q_row[0]:6.3f} | {q_row[1]:6.3f} | "
            f"{q_row[2]:6.3f} | {q_row[3]:6.3f} | "
            f"{ACTION_NAMES[ACTIONS[best_index]]}"
        )


def main() -> None:
    episodes = 1000
    evaluation_episodes = 20
    print(
        "训练参数: "
        "episodes=1000 epsilon=0.30 learning_rate=0.20 "
        "discount_factor=0.95 seed=0"
    )

    try:
        q_table, training_successes = train_q_table(episodes=episodes)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        print("请阅读本文件顶部说明，并且只修改 learn_from_transition()。")
        return

    print(f"训练期间到达终点: {training_successes}/{episodes}")
    print_q_table(q_table)

    q_table_before_evaluation = [row.copy() for row in q_table]
    successes, average_steps, first_path = evaluate_q_table(
        q_table, episodes=evaluation_episodes
    )

    print("\n关闭探索、停止更新后的评估")
    print(f"到达终点: {successes}/{evaluation_episodes}")
    print(f"平均步数: {average_steps:.2f}")
    print("首局路径: " + " -> ".join(map(str, first_path)))
    print(f"评估是否修改 Q 表: {q_table != q_table_before_evaluation}")

    if successes == evaluation_episodes and q_table == q_table_before_evaluation:
        print("练习通过：Q 表通过奖励更新学出了 FrozenLake 路线")


if __name__ == "__main__":
    main()
