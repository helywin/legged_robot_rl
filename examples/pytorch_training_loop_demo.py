#!/usr/bin/env python3
"""把 PyTorch 的一次参数更新重复成最小训练循环。"""

from __future__ import annotations

import torch

from examples.pytorch_module_prediction import OneInputQModule


def train_fixed_sample(
    model: OneInputQModule,
    observation: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
    steps: int,
) -> list[float]:
    """在同一条固定数据上训练若干步，并返回每步更新前损失。"""
    if steps <= 0:
        raise ValueError("steps 必须大于 0")

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        prediction = model(observation)
        loss = (prediction - target) ** 2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def run_demo() -> None:
    model = OneInputQModule(weight=1.0, bias=0.0)
    observation = torch.tensor(0.5, dtype=torch.float32)
    target = torch.tensor(1.0, dtype=torch.float32)
    losses = train_fixed_sample(
        model,
        observation,
        target,
        learning_rate=0.1,
        steps=8,
    )

    for step, loss in enumerate(losses, start=1):
        print(f"step={step:02d} loss_before_update={loss:.6f}")

    with torch.no_grad():
        final_prediction = model(observation).item()
        final_loss = (final_prediction - target.item()) ** 2
    print(f"final_prediction={final_prediction:.6f}")
    print(f"final_loss={final_loss:.6f}")
    print(f"loss_kept_decreasing={all(a > b for a, b in zip(losses, losses[1:]))}")


if __name__ == "__main__":
    run_demo()
