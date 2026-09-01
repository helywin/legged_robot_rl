#!/usr/bin/env python3
"""演示一个共享参数怎样根据 Q 值预测误差逐步更新。"""

from __future__ import annotations

from collections.abc import Sequence


UpdateRecord = tuple[float, float, float]


def predict_q(observation: float, weight: float) -> float:
    """用一个参数估计当前观察下所选动作的 Q 值。"""
    return observation * weight


def calculate_prediction_error(target_q: float, predicted_q: float) -> float:
    """正数表示预测偏低，负数表示预测偏高。"""
    return target_q - predicted_q


def update_weight(
    weight: float,
    observation: float,
    error: float,
    learning_rate: float,
) -> float:
    """让一个线性参数沿着能缩小当前误差的方向移动一小步。"""
    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError("learning_rate 必须在 0 到 1 之间")
    return weight + learning_rate * error * observation


def run_updates(
    observation: float,
    target_q: float,
    learning_rate: float,
    initial_weight: float,
    updates: int,
) -> list[UpdateRecord]:
    """重复更新并返回每次更新前的参数、预测和误差。"""
    if updates < 1:
        raise ValueError("updates 必须至少为 1")

    records = []
    weight = initial_weight
    for _ in range(updates):
        predicted_q = predict_q(observation, weight)
        error = calculate_prediction_error(target_q, predicted_q)
        records.append((weight, predicted_q, error))
        weight = update_weight(weight, observation, error, learning_rate)
    return records


def print_records(records: Sequence[UpdateRecord]) -> None:
    """打印参数更新过程。"""
    for index, (weight, predicted_q, error) in enumerate(records):
        print(
            f"更新前{index}: weight={weight:.6f} "
            f"prediction={predicted_q:.6f} error={error:+.6f}"
        )


if __name__ == "__main__":
    print_records(
        run_updates(
            observation=0.5,
            target_q=1.0,
            learning_rate=0.2,
            initial_weight=1.0,
            updates=6,
        )
    )
