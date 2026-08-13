#!/usr/bin/env python3
"""演示终点价值怎样逐轮传到更早的位置。"""

from __future__ import annotations

import argparse

if __package__:
    from .line_world import RIGHT, LineWorldEnv
    from .q_learning_single_update import calculate_target, update_q_value
    from .train_line_world_q_learning import ACTIONS, QTable, create_q_table
else:
    from line_world import RIGHT, LineWorldEnv
    from q_learning_single_update import calculate_target, update_q_value
    from train_line_world_q_learning import ACTIONS, QTable, create_q_table


RIGHT_INDEX = ACTIONS.index(RIGHT)


def run_always_right_episode(
    q_table: QTable,
    learning_rate: float = 1.0,
    discount_factor: float = 0.9,
) -> None:
    """固定向右走完一轮，并在每个 step 更新对应的 Q 值。"""
    env = LineWorldEnv()
    observation = env.reset()

    while not env.done:
        new_observation, reward, done = env.step(RIGHT)
        best_next_q = 0.0 if done else max(q_table[new_observation])
        target = calculate_target(
            reward, best_next_q, discount_factor, done
        )
        old_q = q_table[observation][RIGHT_INDEX]
        q_table[observation][RIGHT_INDEX] = update_q_value(
            old_q, target, learning_rate
        )
        observation = new_observation


def capture_propagation(
    episodes: int = 4,
    learning_rate: float = 1.0,
    discount_factor: float = 0.9,
) -> list[list[float]]:
    """返回初始状态及每轮结束后各位置 right 的 Q 值。"""
    if episodes < 1:
        raise ValueError("episodes 必须至少为 1")

    q_table = create_q_table(state_count=5, action_count=len(ACTIONS))
    history = [[row[RIGHT_INDEX] for row in q_table[:4]]]

    for _ in range(episodes):
        run_always_right_episode(q_table, learning_rate, discount_factor)
        history.append([row[RIGHT_INDEX] for row in q_table[:4]])

    return history


def run_demo(episodes: int) -> list[list[float]]:
    """打印价值逐轮向前传播的过程。"""
    history = capture_propagation(episodes=episodes)
    print("固定条件: 每轮总是 right, learning_rate=1.0, discount_factor=0.9")
    print("轮次 | 位置0-right | 位置1-right | 位置2-right | 位置3-right")
    print("-----+-------------+-------------+-------------+------------")
    for episode, right_values in enumerate(history):
        label = "初始" if episode == 0 else str(episode)
        values = " | ".join(f"{value:11.4f}" for value in right_values)
        print(f"{label:>4} | {values}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.episodes)
