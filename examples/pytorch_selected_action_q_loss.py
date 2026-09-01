#!/usr/bin/env python3
"""从两个动作 Q 值中选择实际动作对应的值并计算损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


@dataclass(frozen=True)
class SelectedActionResult:
    """记录一次所选动作损失和梯度。"""

    q_values: list[float]
    best_action: int
    executed_action: int
    selected_q: float
    target: float
    loss: float
    weight_gradients: list[list[float]]
    bias_gradients: list[float]


def calculate_selected_action_loss(
    model: TwoActionQModule,
    observation: torch.Tensor,
    executed_action: int,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回全部 Q 值、实际动作 Q 值和平方损失，并计算梯度。"""
    q_values = model(observation)
    selected_q = q_values[executed_action]
    loss = (selected_q - target) ** 2
    loss.backward()
    return q_values, selected_q, loss


def run_demo() -> None:
    model = TwoActionQModule()
    set_demo_parameters(model)
    observation = torch.tensor([0.2, -0.1], dtype=torch.float32)
    executed_action = 1
    target = torch.tensor(0.5, dtype=torch.float32)

    q_values, selected_q, loss = calculate_selected_action_loss(
        model, observation, executed_action, target
    )
    def clean_round(value: float) -> float:
        rounded = round(value, 3)
        return 0.0 if rounded == 0.0 else rounded

    result = SelectedActionResult(
        q_values=[round(value, 3) for value in q_values.tolist()],
        best_action=q_values.argmax().item(),
        executed_action=executed_action,
        selected_q=selected_q.item(),
        target=target.item(),
        loss=loss.item(),
        weight_gradients=[
            [clean_round(value) for value in row]
            for row in model.q_values.weight.grad.tolist()
        ],
        bias_gradients=[
            clean_round(value) for value in model.q_values.bias.grad.tolist()
        ],
    )

    print(f"q_values={result.q_values}")
    print(f"best_action={result.best_action}")
    print(f"executed_action={result.executed_action}")
    print(f"selected_q={result.selected_q:+.3f}")
    print(f"target={result.target:+.3f}")
    print(f"loss={result.loss:+.4f}")
    print(f"weight_gradients={result.weight_gradients}")
    print(f"bias_gradients={result.bias_gradients}")


if __name__ == "__main__":
    run_demo()
