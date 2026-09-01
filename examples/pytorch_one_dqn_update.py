#!/usr/bin/env python3
"""把一条经验的在线预测、目标和优化器组成一次 PyTorch DQN 更新。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from examples.pytorch_dqn_target_q import calculate_dqn_target
from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


@dataclass(frozen=True)
class DqnUpdateResult:
    """记录一次更新前后的关键数字。"""

    q_values_before: list[float]
    selected_q_before: float
    target_q: float
    loss: float
    selected_q_after: float


def update_from_one_transition(
    online_network: TwoActionQModule,
    target_network: TwoActionQModule,
    optimizer: torch.optim.Optimizer,
    observation: torch.Tensor,
    executed_action: int,
    reward: torch.Tensor,
    next_observation: torch.Tensor,
    discount_factor: float,
    terminated: bool,
) -> DqnUpdateResult:
    """使用一条经验更新一次在线网络，目标网络保持不变。"""
    optimizer.zero_grad()
    q_values = online_network(observation)
    selected_q = q_values[executed_action]
    target_q = calculate_dqn_target(
        target_network,
        next_observation,
        reward,
        discount_factor,
        terminated,
    )
    loss = (selected_q - target_q) ** 2
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        selected_q_after = online_network(observation)[executed_action]

    return DqnUpdateResult(
        q_values_before=[round(value, 4) for value in q_values.tolist()],
        selected_q_before=selected_q.item(),
        target_q=target_q.item(),
        loss=loss.item(),
        selected_q_after=selected_q_after.item(),
    )


def run_demo() -> None:
    online_network = TwoActionQModule()
    target_network = TwoActionQModule()
    set_demo_parameters(online_network)
    set_demo_parameters(target_network)
    optimizer = torch.optim.SGD(online_network.parameters(), lr=0.1)
    observation = torch.tensor([0.2, -0.1], dtype=torch.float32)
    next_observation = torch.tensor([0.4, 0.2], dtype=torch.float32)
    reward = torch.tensor(0.2, dtype=torch.float32)

    online_before = [
        parameter.detach().clone()
        for parameter in online_network.parameters()
    ]
    target_before = [
        parameter.detach().clone()
        for parameter in target_network.parameters()
    ]
    result = update_from_one_transition(
        online_network,
        target_network,
        optimizer,
        observation,
        executed_action=1,
        reward=reward,
        next_observation=next_observation,
        discount_factor=0.9,
        terminated=False,
    )

    print(f"q_values_before={result.q_values_before}")
    print(f"selected_q_before={result.selected_q_before:+.4f}")
    print(f"target_q={result.target_q:+.4f}")
    print(f"loss={result.loss:+.4f}")
    print(f"selected_q_after={result.selected_q_after:+.4f}")
    print(
        "selected_q_moved_toward_target="
        f"{abs(result.selected_q_after - result.target_q) < abs(result.selected_q_before - result.target_q)}"
    )
    print(
        "unselected_online_row_unchanged="
        f"{torch.equal(online_before[0][0], online_network.q_values.weight[0])}"
    )
    print(
        "target_parameters_unchanged="
        f"{all(torch.equal(old, current) for old, current in zip(target_before, target_network.parameters()))}"
    )


if __name__ == "__main__":
    run_demo()
