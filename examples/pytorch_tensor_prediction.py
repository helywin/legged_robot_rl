#!/usr/bin/env python3
"""把普通 Python 数字换成 PyTorch 张量，完成同一个最小预测。"""

from __future__ import annotations

import torch


def predict_q(
    observation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """使用三个标量张量返回一个标量预测张量。"""
    weighted_observation = observation * weight
    prediction = weighted_observation + bias
    return prediction


def run_demo() -> None:
    observation = torch.tensor(-0.02, dtype=torch.float32)
    weight = torch.tensor(3.0, dtype=torch.float32)
    bias = torch.tensor(0.1, dtype=torch.float32)
    prediction = predict_q(observation, weight, bias)

    print(f"observation={observation.item():+.2f}")
    print(f"weight={weight.item():+.2f}")
    print(f"bias={bias.item():+.2f}")
    print(f"prediction={prediction.item():+.2f}")
    print(f"prediction_type={type(prediction).__name__}")
    print(f"prediction_dtype={prediction.dtype}")
    print(f"prediction_dimensions={prediction.ndim}")


if __name__ == "__main__":
    run_demo()
