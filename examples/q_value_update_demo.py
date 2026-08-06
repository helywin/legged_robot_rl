#!/usr/bin/env python3
"""演示 Q 值如何根据一次即时奖励做小幅更新。"""

from __future__ import annotations

import argparse


def update_q_value(old_q: float, reward: float, learning_rate: float) -> float:
    """让旧 Q 值朝本次奖励移动 learning_rate 指定的比例。"""
    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError("learning_rate 必须在 0 到 1 之间")
    gap = reward - old_q
    return old_q + learning_rate * gap


def run_demo(
    initial_q: float, reward: float, learning_rate: float, updates: int
) -> float:
    """重复收到相同奖励并打印每一次 Q 值更新。"""
    if updates < 1:
        raise ValueError("updates 必须至少为 1")

    q_value = initial_q
    print(
        f"初始 Q={q_value:.4f}, reward={reward:.4f}, "
        f"learning_rate={learning_rate:.2f}"
    )

    for attempt in range(1, updates + 1):
        gap = reward - q_value
        adjustment = learning_rate * gap
        new_q = update_q_value(q_value, reward, learning_rate)
        print(
            f"第 {attempt} 次: 差距={gap:+.4f}, "
            f"本次修正={adjustment:+.4f}, 新 Q={new_q:.4f}"
        )
        q_value = new_q

    return q_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-q", type=float, default=0.3)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--updates", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.initial_q, args.reward, args.learning_rate, args.updates)
