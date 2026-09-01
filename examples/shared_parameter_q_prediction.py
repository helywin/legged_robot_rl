#!/usr/bin/env python3
"""用共享参数根据观察估计两个离散动作的 Q 值。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


LEFT = "LEFT"
RIGHT = "RIGHT"
ACTIONS = (LEFT, RIGHT)

Observation = tuple[float, float]
ActionParameters = tuple[float, float, float]
Parameters = Mapping[str, ActionParameters]

DEFAULT_PARAMETERS: dict[str, ActionParameters] = {
    LEFT: (-0.4, 0.8, 0.1),
    RIGHT: (0.7, -0.5, 0.0),
}


def predict_q_values(
    observation: Observation, parameters: Parameters
) -> list[float]:
    """返回顺序与 ACTIONS 一致的两个动作 Q 值估计。"""
    position, danger = observation
    q_values = []
    for action in ACTIONS:
        position_weight, danger_weight, bias = parameters[action]
        q_values.append(
            position * position_weight + danger * danger_weight + bias
        )
    return q_values


def choose_best_action(q_values: Sequence[float]) -> str:
    """选择 Q 值最大的动作；并列时选择 ACTIONS 中更靠前的动作。"""
    if len(q_values) != len(ACTIONS):
        raise ValueError(f"q_values 必须包含 {len(ACTIONS)} 个值")
    best_index = max(range(len(ACTIONS)), key=q_values.__getitem__)
    return ACTIONS[best_index]


def run_demo() -> None:
    """保持参数不变，只改变观察并打印预测结果。"""
    observations = ((0.2, 0.1), (0.8, 0.1), (0.8, 0.9))
    parameters_before = dict(DEFAULT_PARAMETERS)

    for observation in observations:
        q_values = predict_q_values(observation, DEFAULT_PARAMETERS)
        rounded = [round(value, 3) for value in q_values]
        print(
            f"observation={observation} "
            f"q_values={rounded} "
            f"best={choose_best_action(q_values)}"
        )

    print(f"parameters_changed={DEFAULT_PARAMETERS != parameters_before}")


if __name__ == "__main__":
    run_demo()
