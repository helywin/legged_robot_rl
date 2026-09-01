#!/usr/bin/env python3
"""用 nn.Module 组织一个输入、一个输出的最小 PyTorch 模型。"""

from __future__ import annotations

import torch
from torch import nn


class OneInputQModule(nn.Module):
    """登记一个权重和一个偏置，并在 forward 中完成预测。"""

    def __init__(self, weight: float, bias: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """给 VS Code/Pylance 明确调用类型，并保留 Module 调用流程。"""
        return super().__call__(observation)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """PyTorch 调用模型对象时自动执行这里。"""
        return observation * self.weight + self.bias


def run_demo() -> None:
    model = OneInputQModule(weight=3.0, bias=0.1)
    observation = torch.tensor(-0.02, dtype=torch.float32)
    prediction = model(observation)

    print(f"is_nn_module={isinstance(model, nn.Module)}")
    for name, parameter in model.named_parameters():
        print(f"registered_parameter={name} value={parameter.item():+.2f}")
    print(f"observation={observation.item():+.2f}")
    print(f"prediction={prediction.item():+.2f}")
    print(f"prediction_type={type(prediction).__name__}")


if __name__ == "__main__":
    run_demo()
