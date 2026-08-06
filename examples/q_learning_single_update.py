#!/usr/bin/env python3
"""把未来目标和学习率拼成一次完整的 Q-learning 更新。"""

from __future__ import annotations

import argparse


def calculate_target(
    reward: float, best_next_q: float, discount_factor: float, done: bool
) -> float:
    """计算本次更新要参考的目标值。"""
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor 必须在 0 到 1 之间")
    if done:
        return reward
    return reward + discount_factor * best_next_q


def update_q_value(
    old_q: float,
    target: float,
    learning_rate: float,
) -> float:
    """让旧 Q 值按学习率朝目标值移动。"""
    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError("learning_rate 必须在 0 到 1 之间")
    return old_q + learning_rate * (target - old_q)


def run_demo(
    old_q: float,
    reward: float,
    best_next_q: float,
    discount_factor: float,
    learning_rate: float,
    done: bool,
) -> tuple[float, float]:
    """打印一次 Q-learning 更新的两个计算步骤。"""
    target = calculate_target(reward, best_next_q, discount_factor, done)
    gap = target - old_q
    adjustment = learning_rate * gap
    new_q = update_q_value(old_q, target, learning_rate)

    print(f"旧 Q 值={old_q:.4f}")
    if done:
        print(f"已到终点: 目标值=当前奖励={target:.4f}")
    else:
        print(
            f"目标值={reward:.4f} + {discount_factor:.2f} × "
            f"{best_next_q:.4f} = {target:.4f}"
        )
    print(f"差距={target:.4f} - {old_q:.4f} = {gap:.4f}")
    print(
        f"本次修正={learning_rate:.2f} × {gap:.4f} = {adjustment:.4f}"
    )
    print(f"新 Q 值={old_q:.4f} + {adjustment:.4f} = {new_q:.4f}")
    return target, new_q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-q", type=float, default=0.3)
    parser.add_argument("--reward", type=float, default=-0.01)
    parser.add_argument("--best-next-q", type=float, default=0.8)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--done", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        args.old_q,
        args.reward,
        args.best_next_q,
        args.discount_factor,
        args.learning_rate,
        args.done,
    )
