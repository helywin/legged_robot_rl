#!/usr/bin/env python3
"""比较第二次更新前清空和不清空旧梯度的差异。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from examples.pytorch_module_prediction import OneInputQModule


@dataclass(frozen=True)
class SecondUpdateResult:
    """记录第二个目标更新前后的关键数字。"""

    prediction_before: float
    loss_before: float
    weight_gradient: float
    bias_gradient: float
    prediction_after: float
    loss_after: float
    moved_toward_target: bool


def prepare_first_update() -> tuple[OneInputQModule, torch.optim.Optimizer]:
    """完成第一次更新，并有意把旧梯度留在参数的 grad 中。"""
    model = OneInputQModule(weight=1.0, bias=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    observation = torch.tensor(0.5, dtype=torch.float32)
    first_target = torch.tensor(1.0, dtype=torch.float32)

    first_prediction = model(observation)
    first_loss = (first_prediction - first_target) ** 2
    first_loss.backward()
    optimizer.step()
    return model, optimizer


def update_on_second_target(clear_old_gradient: bool) -> SecondUpdateResult:
    """使用与第一个目标方向相反的第二个目标更新一次。"""
    model, optimizer = prepare_first_update()
    observation = torch.tensor(0.5, dtype=torch.float32)
    second_target = torch.tensor(0.4, dtype=torch.float32)

    if clear_old_gradient:
        optimizer.zero_grad()

    prediction_before = model(observation)
    loss_before = (prediction_before - second_target) ** 2
    loss_before.backward()
    weight_gradient = model.weight.grad.item()
    bias_gradient = model.bias.grad.item()
    optimizer.step()

    with torch.no_grad():
        prediction_after = model(observation)
        loss_after = (prediction_after - second_target) ** 2

    return SecondUpdateResult(
        prediction_before=prediction_before.item(),
        loss_before=loss_before.item(),
        weight_gradient=weight_gradient,
        bias_gradient=bias_gradient,
        prediction_after=prediction_after.item(),
        loss_after=loss_after.item(),
        moved_toward_target=(
            abs(prediction_after.item() - second_target.item())
            < abs(prediction_before.item() - second_target.item())
        ),
    )


def run_demo() -> None:
    without_clear = update_on_second_target(clear_old_gradient=False)
    with_clear = update_on_second_target(clear_old_gradient=True)

    print("without_zero_grad")
    print(f"  weight_gradient={without_clear.weight_gradient:+.3f}")
    print(f"  prediction_after={without_clear.prediction_after:+.3f}")
    print(f"  loss_after={without_clear.loss_after:+.3f}")
    print(f"  moved_toward_target={without_clear.moved_toward_target}")
    print("with_zero_grad")
    print(f"  weight_gradient={with_clear.weight_gradient:+.3f}")
    print(f"  prediction_after={with_clear.prediction_after:+.3f}")
    print(f"  loss_after={with_clear.loss_after:+.3f}")
    print(f"  moved_toward_target={with_clear.moved_toward_target}")


if __name__ == "__main__":
    run_demo()
