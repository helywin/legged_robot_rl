#!/usr/bin/env python3
"""让 SGD 优化器使用已有梯度完成一次参数更新。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from examples.pytorch_module_prediction import OneInputQModule


@dataclass(frozen=True)
class UpdateResult:
    """保存一次更新前后的可见结果。"""

    prediction_before: float
    loss_before: float
    weight_gradient: float
    bias_gradient: float
    weight_after: float
    bias_after: float
    prediction_after: float
    loss_after: float


def make_one_optimizer_step(
    model: OneInputQModule,
    observation: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
) -> UpdateResult:
    """计算一次梯度，调用一次 SGD step，并返回更新前后结果。"""
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    prediction_before = model(observation)
    loss_before = (prediction_before - target) ** 2
    loss_before.backward()

    weight_gradient = model.weight.grad.item()
    bias_gradient = model.bias.grad.item()
    optimizer.step()

    with torch.no_grad():
        prediction_after = model(observation)
        loss_after = (prediction_after - target) ** 2

    return UpdateResult(
        prediction_before=prediction_before.item(),
        loss_before=loss_before.item(),
        weight_gradient=weight_gradient,
        bias_gradient=bias_gradient,
        weight_after=model.weight.item(),
        bias_after=model.bias.item(),
        prediction_after=prediction_after.item(),
        loss_after=loss_after.item(),
    )


def run_demo() -> None:
    model = OneInputQModule(weight=1.0, bias=0.0)
    observation = torch.tensor(0.5, dtype=torch.float32)
    target = torch.tensor(1.0, dtype=torch.float32)
    result = make_one_optimizer_step(
        model, observation, target, learning_rate=0.1
    )

    print(f"prediction_before={result.prediction_before:+.3f}")
    print(f"loss_before={result.loss_before:+.3f}")
    print(f"weight_gradient={result.weight_gradient:+.3f}")
    print(f"bias_gradient={result.bias_gradient:+.3f}")
    print(f"weight_after={result.weight_after:+.3f}")
    print(f"bias_after={result.bias_after:+.3f}")
    print(f"prediction_after={result.prediction_after:+.3f}")
    print(f"loss_after={result.loss_after:+.3f}")
    print(f"loss_decreased={result.loss_after < result.loss_before}")


if __name__ == "__main__":
    run_demo()
