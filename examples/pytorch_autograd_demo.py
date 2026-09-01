#!/usr/bin/env python3
"""计算预测损失并用 backward() 得到参数梯度，但不更新参数。"""

from __future__ import annotations

import torch

from examples.pytorch_module_prediction import OneInputQModule


def calculate_loss_and_gradients(
    model: OneInputQModule,
    observation: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """完成预测、平方损失和反向计算，返回预测与损失。"""
    prediction = model(observation)
    loss = (prediction - target) ** 2
    loss.backward()
    return prediction, loss


def run_demo() -> None:
    model = OneInputQModule(weight=1.0, bias=0.0)
    observation = torch.tensor(0.5, dtype=torch.float32)
    target = torch.tensor(1.0, dtype=torch.float32)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    prediction, loss = calculate_loss_and_gradients(model, observation, target)
    unchanged = all(
        torch.equal(old, current)
        for old, current in zip(parameters_before, model.parameters())
    )

    print(f"observation={observation.item():+.2f}")
    print(f"prediction={prediction.item():+.2f}")
    print(f"target={target.item():+.2f}")
    print(f"loss={loss.item():+.2f}")
    print(f"weight_grad={model.weight.grad.item():+.2f}")
    print(f"bias_grad={model.bias.grad.item():+.2f}")
    print(f"parameters_changed={not unchanged}")


if __name__ == "__main__":
    run_demo()
