#!/usr/bin/env python3
"""用纯 Python Q-learning 训练走格子策略。"""

from __future__ import annotations

import argparse
import random

if __package__:
    from .line_world import LEFT, RIGHT, LineWorldEnv
    from .q_learning_single_update import calculate_target, update_q_value
else:
    from line_world import LEFT, RIGHT, LineWorldEnv
    from q_learning_single_update import calculate_target, update_q_value


ACTIONS = (LEFT, RIGHT)
QTable = list[list[float]]


def create_q_table(state_count: int, action_count: int) -> QTable:
    """创建全零 Q 表：每行是观察，每列是动作。"""
    if state_count < 1 or action_count < 1:
        raise ValueError("状态数和动作数必须至少为 1")
    return [[0.0 for _ in range(action_count)] for _ in range(state_count)]


def choose_action_index(
    q_row: list[float], epsilon: float, rng: random.Random
) -> int:
    """使用 epsilon-greedy 从当前观察对应的一行中选择动作。"""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon 必须在 0 到 1 之间")

    if rng.random() < epsilon:
        return rng.randrange(len(q_row))

    best_value = max(q_row)
    best_indices = [
        action_index
        for action_index, q_value in enumerate(q_row)
        if q_value == best_value
    ]
    return rng.choice(best_indices)


def train_q_table(
    episodes: int = 500,
    epsilon: float = 0.2,
    learning_rate: float = 0.2,
    discount_factor: float = 0.9,
    seed: int = 0,
) -> tuple[QTable, int]:
    """训练多轮并返回 Q 表和训练期间的到达目标次数。"""
    if episodes < 1:
        raise ValueError("episodes 必须至少为 1")

    env = LineWorldEnv()
    q_table = create_q_table(env.length, len(ACTIONS))
    rng = random.Random(seed)
    successes = 0

    for _ in range(episodes):
        observation = env.reset()

        while not env.done:
            action_index = choose_action_index(
                q_table[observation], epsilon, rng
            )
            action = ACTIONS[action_index]
            new_observation, reward, done = env.step(action)

            best_next_q = 0.0 if done else max(q_table[new_observation])
            target = calculate_target(
                reward, best_next_q, discount_factor, done
            )
            old_q = q_table[observation][action_index]
            q_table[observation][action_index] = update_q_value(
                old_q, target, learning_rate
            )

            observation = new_observation

        if env.position == env.length - 1:
            successes += 1

    return q_table, successes


def evaluate_q_table(
    q_table: QTable, episodes: int = 10
) -> tuple[int, float, float]:
    """关闭探索且不更新 Q 表，评估当前策略。"""
    if episodes < 1:
        raise ValueError("episodes 必须至少为 1")

    successes = 0
    total_steps = 0
    total_reward = 0.0

    for _ in range(episodes):
        env = LineWorldEnv()
        observation = env.reset()

        while not env.done:
            action_index = max(
                range(len(ACTIONS)),
                key=q_table[observation].__getitem__,
            )
            observation, reward, _ = env.step(ACTIONS[action_index])
            total_reward += reward

        total_steps += env.step_count
        if env.position == env.length - 1:
            successes += 1

    return successes, total_steps / episodes, total_reward / episodes


def print_q_table(q_table: QTable) -> None:
    """用便于初学者阅读的格式打印 Q 表。"""
    print("\n训练后的 Q 表")
    print("位置 |   left |  right | 当前最佳动作")
    print("-----+--------+--------+-------------")
    for observation, q_row in enumerate(q_table):
        if observation == len(q_table) - 1:
            best_action = "终点"
        else:
            best_index = max(range(len(ACTIONS)), key=q_row.__getitem__)
            best_action = ACTIONS[best_index]
        print(
            f"{observation:4d} | {q_row[0]:6.3f} | "
            f"{q_row[1]:6.3f} | {best_action}"
        )


def run_demo(
    episodes: int,
    epsilon: float,
    learning_rate: float,
    discount_factor: float,
    seed: int,
    evaluation_episodes: int,
) -> QTable:
    """训练、打印 Q 表，再关闭探索进行评估。"""
    q_table, training_successes = train_q_table(
        episodes,
        epsilon,
        learning_rate,
        discount_factor,
        seed,
    )
    print(
        f"训练: episodes={episodes}, epsilon={epsilon:.2f}, "
        f"learning_rate={learning_rate:.2f}, "
        f"discount_factor={discount_factor:.2f}, seed={seed}"
    )
    print(f"训练期间到达目标: {training_successes}/{episodes}")
    print_q_table(q_table)

    successes, average_steps, average_reward = evaluate_q_table(
        q_table, evaluation_episodes
    )
    print("\n关闭探索并停止更新后的评估")
    print(f"到达目标: {successes}/{evaluation_episodes}")
    print(f"平均步数: {average_steps:.2f}")
    print(f"平均总奖励: {average_reward:+.2f}")
    return q_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        args.episodes,
        args.epsilon,
        args.learning_rate,
        args.discount_factor,
        args.seed,
        args.evaluation_episodes,
    )
