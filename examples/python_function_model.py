#!/usr/bin/env python3
"""用普通 Python 函数演示最小的“输入经过参数计算得到输出”。"""

from __future__ import annotations


def predict_q(observation: float, weight: float, bias: float) -> float:
    """根据一项观察、一个权重和一个偏置返回预测 Q 值。"""
    weighted_observation = observation * weight
    prediction = weighted_observation + bias
    return prediction


def run_demo() -> None:
    """把函数接收的值和每一步计算直接打印出来。"""
    observation = -0.02
    weight = 3.0
    bias = 0.1
    weighted_observation = observation * weight
    prediction = predict_q(observation, weight, bias)

    print(f"observation={observation:+.2f}")
    print(f"weight={weight:+.2f}")
    print(f"weighted_observation={weighted_observation:+.2f}")
    print(f"bias={bias:+.2f}")
    print(f"prediction={prediction:+.2f}")


if __name__ == "__main__":
    run_demo()
