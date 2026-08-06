#!/usr/bin/env python3
"""用固定 Q 分数演示 epsilon-greedy 如何选择动作。"""

from __future__ import annotations

import argparse
import random


LEFT = "left"
RIGHT = "right"
ACTIONS = (LEFT, RIGHT)
Q_VALUES = {LEFT: 0.30, RIGHT: 0.80}
EXPLORE = "探索"
EXPLOIT = "利用"


def choose_action(
    q_values: dict[str, float], epsilon: float, rng: random.Random
) -> tuple[str, str]:
    """返回动作和本次选择方式。"""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon 必须在 0 到 1 之间")

    if rng.random() < epsilon:
        return rng.choice(ACTIONS), EXPLORE

    best_action = max(ACTIONS, key=q_values.__getitem__)
    return best_action, EXPLOIT


def run_demo(epsilon: float, decisions: int, seed: int) -> dict[str, int]:
    """重复选择动作并统计结果；本示例不会更新 Q 分数。"""
    if decisions < 1:
        raise ValueError("decisions 必须至少为 1")

    rng = random.Random(seed)
    counts = {EXPLORE: 0, EXPLOIT: 0, LEFT: 0, RIGHT: 0}

    for _ in range(decisions):
        action, mode = choose_action(Q_VALUES, epsilon, rng)
        counts[mode] += 1
        counts[action] += 1

    print(f"固定 Q 分数: left={Q_VALUES[LEFT]:.2f}, right={Q_VALUES[RIGHT]:.2f}")
    print(f"epsilon={epsilon:.2f}, 决策次数={decisions}, seed={seed}")
    print(f"选择方式: 探索={counts[EXPLORE]}, 利用={counts[EXPLOIT]}")
    print(f"最终动作: left={counts[LEFT]}, right={counts[RIGHT]}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--decisions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.epsilon, args.decisions, args.seed)
