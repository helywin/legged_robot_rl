#!/usr/bin/env python3
"""用一个 PyTorch 线性层把两个观察量变成两个动作 Q 值。"""

from __future__ import annotations

import torch
from torch import nn


class TwoActionQModule(nn.Module):
    """接收两个观察量，按固定动作顺序输出两个 Q 值。"""

    def __init__(self) -> None:
        super().__init__()
        self.q_values = nn.Linear(in_features=2, out_features=2)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """向 VS Code/Pylance 暴露明确的张量输入输出类型。"""
        return super().__call__(observation)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """返回顺序为 [LEFT, RIGHT] 的两个 Q 值。"""
        return self.q_values(observation)


def set_demo_parameters(model: TwoActionQModule) -> None:
    """设置固定参数，便于手算并复现实例输出。"""
    with torch.no_grad():
        model.q_values.weight.copy_(
            torch.tensor(
                [[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32
            )
        )
        model.q_values.bias.copy_(
            torch.tensor([0.1, 0.2], dtype=torch.float32)
        )


def run_demo() -> None:
    model = TwoActionQModule()
    set_demo_parameters(model)
    observations = (
        torch.tensor([0.2, -0.1]),
        torch.tensor([-1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    )

    print(f"weight_shape={tuple(model.q_values.weight.shape)}")
    print(f"bias_shape={tuple(model.q_values.bias.shape)}")
    for observation in observations:
        q_values = model(observation)
        best_action = q_values.argmax().item()
        shown_observation = [
            round(value, 3) for value in observation.tolist()
        ]
        print(
            f"observation={shown_observation} "
            f"q_values={[round(value, 3) for value in q_values.tolist()]} "
            f"best_action={best_action}"
        )


if __name__ == "__main__":
    run_demo()
