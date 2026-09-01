#!/usr/bin/env python3
"""串起一条回放经验在简化 DQN 更新中的完整数据流。"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .replay_buffer_demo import ReplayBuffer, Transition


LEFT = "LEFT"
RIGHT = "RIGHT"
ACTIONS = (LEFT, RIGHT)
ActionWeights = dict[str, float]


@dataclass(frozen=True)
class TrainingStep:
    """一条回放经验完成更新时的关键中间值。"""

    transition: Transition
    predicted_q_values: dict[str, float]
    selected_prediction: float
    next_target_q_values: dict[str, float]
    target_q: float
    error: float
    old_weight: float
    new_weight: float


def predict_q_values(
    observation: float,
    action_weights: ActionWeights,
) -> dict[str, float]:
    """为每个离散动作计算一个简化的线性 Q 值。"""
    return {
        action: observation * action_weights[action] for action in ACTIONS
    }


def calculate_target_q(
    transition: Transition,
    target_weights: ActionWeights,
    discount_factor: float,
) -> tuple[float, dict[str, float]]:
    """用目标参数计算目标 Q 值，并返回下一观察的动作 Q 值。"""
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor 必须在 0 到 1 之间")

    next_q_values = predict_q_values(
        float(transition.new_observation), target_weights
    )
    if transition.terminated:
        return transition.reward, next_q_values
    best_future_q = max(next_q_values.values())
    return transition.reward + discount_factor * best_future_q, next_q_values


def train_one_transition(
    transition: Transition,
    online_weights: ActionWeights,
    target_weights: ActionWeights,
    learning_rate: float,
    discount_factor: float,
) -> tuple[ActionWeights, TrainingStep]:
    """用一条回放经验更新在线参数，保持目标参数不变。"""
    if transition.action not in ACTIONS:
        raise ValueError(f"未知动作：{transition.action}")
    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError("learning_rate 必须在 0 到 1 之间")

    predicted_q_values = predict_q_values(
        float(transition.observation), online_weights
    )
    selected_prediction = predicted_q_values[transition.action]
    target_q, next_target_q_values = calculate_target_q(
        transition, target_weights, discount_factor
    )
    error = target_q - selected_prediction

    old_weight = online_weights[transition.action]
    new_weight = (
        old_weight
        + learning_rate * error * float(transition.observation)
    )
    updated_online_weights = dict(online_weights)
    updated_online_weights[transition.action] = new_weight

    return updated_online_weights, TrainingStep(
        transition=transition,
        predicted_q_values=predicted_q_values,
        selected_prediction=selected_prediction,
        next_target_q_values=next_target_q_values,
        target_q=target_q,
        error=error,
        old_weight=old_weight,
        new_weight=new_weight,
    )


def make_demo_buffer() -> ReplayBuffer:
    """创建包含不同时刻经验的演示缓冲区。"""
    replay_buffer = ReplayBuffer(capacity=4)
    replay_buffer.add(Transition(1, RIGHT, 0.0, 2, False, False))
    replay_buffer.add(Transition(2, LEFT, 0.1, 3, False, False))
    replay_buffer.add(Transition(3, RIGHT, 0.2, 4, False, False))
    return replay_buffer


def rounded_q_values(q_values: dict[str, float]) -> dict[str, float]:
    """把终端演示中的浮点数缩短到三位小数。"""
    return {action: round(value, 3) for action, value in q_values.items()}


def run_demo() -> None:
    """固定随机种子抽取一条旧经验并完成一次简化更新。"""
    replay_buffer = make_demo_buffer()
    sampled = replay_buffer.sample(1, random.Random(7))[0]
    online_weights = {LEFT: 0.2, RIGHT: 0.4}
    target_weights = {LEFT: 0.3, RIGHT: 0.5}
    target_before = dict(target_weights)

    updated_online, step = train_one_transition(
        transition=sampled,
        online_weights=online_weights,
        target_weights=target_weights,
        learning_rate=0.2,
        discount_factor=0.9,
    )

    print(
        "buffer_observations="
        + str([item.observation for item in replay_buffer.snapshot()])
    )
    print(
        f"sampled=(observation={sampled.observation}, "
        f"action={sampled.action}, reward={sampled.reward}, "
        f"new_observation={sampled.new_observation})"
    )
    print(f"online_q_values={rounded_q_values(step.predicted_q_values)}")
    print(f"selected_prediction={step.selected_prediction:.6f}")
    print(
        "next_target_q_values="
        + str(rounded_q_values(step.next_target_q_values))
    )
    print(f"target_q={step.target_q:.6f}")
    print(f"error={step.error:+.6f}")
    print(
        f"updated_{sampled.action}_weight="
        f"{step.old_weight:.6f}->{step.new_weight:.6f}"
    )
    other_action = LEFT if sampled.action == RIGHT else RIGHT
    print(
        f"other_online_weight_unchanged="
        f"{updated_online[other_action] == online_weights[other_action]}"
    )
    print(f"target_weights_changed={target_weights != target_before}")


if __name__ == "__main__":
    run_demo()
