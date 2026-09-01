#!/usr/bin/env python3
"""用两个单参数估计器演示 DQN 目标网络的固定与定期同步。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UpdateRecord:
    """一次在线参数更新前后的可观察数据。"""

    update_index: int
    online_before: float
    target_before: float
    target_q: float
    error: float
    online_after: float
    target_after: float
    synced: bool


def predict_q(observation: float, weight: float) -> float:
    """用一个参数估计 Q 值。"""
    return observation * weight


def calculate_target_q(
    reward: float,
    new_observation: float,
    target_weight: float,
    discount_factor: float,
    terminated: bool,
) -> float:
    """使用暂时固定的目标参数计算 Q-learning 目标。"""
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor 必须在 0 到 1 之间")
    if terminated:
        return reward
    future_q = predict_q(new_observation, target_weight)
    return reward + discount_factor * future_q


def update_online_weight(
    online_weight: float,
    observation: float,
    error: float,
    learning_rate: float,
) -> float:
    """只更新在线参数，不直接修改目标参数。"""
    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError("learning_rate 必须在 0 到 1 之间")
    return online_weight + learning_rate * error * observation


def run_updates(
    updates: int = 6,
    sync_interval: int = 3,
) -> list[UpdateRecord]:
    """固定其他条件，观察在线参数更新和目标参数定期同步。"""
    if updates < 1:
        raise ValueError("updates 必须至少为 1")
    if sync_interval < 1:
        raise ValueError("sync_interval 必须至少为 1")

    observation = 1.0
    new_observation = 1.0
    reward = 0.2
    discount_factor = 0.9
    learning_rate = 0.2
    online_weight = 1.0
    target_weight = online_weight
    records = []

    for update_index in range(1, updates + 1):
        online_before = online_weight
        target_before = target_weight
        predicted_q = predict_q(observation, online_weight)
        target_q = calculate_target_q(
            reward,
            new_observation,
            target_weight,
            discount_factor,
            terminated=False,
        )
        error = target_q - predicted_q
        online_weight = update_online_weight(
            online_weight,
            observation,
            error,
            learning_rate,
        )

        synced = update_index % sync_interval == 0
        if synced:
            target_weight = online_weight

        records.append(
            UpdateRecord(
                update_index=update_index,
                online_before=online_before,
                target_before=target_before,
                target_q=target_q,
                error=error,
                online_after=online_weight,
                target_after=target_weight,
                synced=synced,
            )
        )

    return records


def print_records(records: list[UpdateRecord]) -> None:
    """打印目标保持固定以及同步发生的时刻。"""
    for record in records:
        print(
            f"step={record.update_index} "
            f"online={record.online_before:.6f}->{record.online_after:.6f} "
            f"target={record.target_before:.6f}->{record.target_after:.6f} "
            f"target_q={record.target_q:.6f} "
            f"synced={record.synced}"
        )


if __name__ == "__main__":
    print_records(run_updates())
