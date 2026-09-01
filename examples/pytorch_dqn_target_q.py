#!/usr/bin/env python3
"""用奖励和目标网络的下一观察估值计算 DQN 目标 Q 值。"""

from __future__ import annotations

import torch

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


def calculate_dqn_target(
    target_network: TwoActionQModule,
    next_observation: torch.Tensor,
    reward: torch.Tensor,
    discount_factor: float,
    terminated: bool,
) -> torch.Tensor:
    """返回不连接梯度的标量 DQN 目标 Q 值。"""
    with torch.no_grad():
        if terminated:
            return reward.clone()
        next_q_values = target_network(next_observation)
        best_next_q = next_q_values.max()
        return reward + discount_factor * best_next_q


def run_demo() -> None:
    target_network = TwoActionQModule()
    set_demo_parameters(target_network)
    next_observation = torch.tensor([0.4, 0.2], dtype=torch.float32)
    reward = torch.tensor(0.2, dtype=torch.float32)

    with torch.no_grad():
        next_q_values = target_network(next_observation)
    continuing_target = calculate_dqn_target(
        target_network,
        next_observation,
        reward,
        discount_factor=0.9,
        terminated=False,
    )
    terminal_target = calculate_dqn_target(
        target_network,
        next_observation,
        reward,
        discount_factor=0.9,
        terminated=True,
    )

    print(
        "next_q_values="
        f"{[round(value, 3) for value in next_q_values.tolist()]}"
    )
    print(f"best_next_q={next_q_values.max().item():+.3f}")
    print(f"reward={reward.item():+.3f}")
    print(f"continuing_target={continuing_target.item():+.3f}")
    print(f"terminal_target={terminal_target.item():+.3f}")
    print(f"target_requires_grad={continuing_target.requires_grad}")
    print(
        "target_parameter_gradients_none="
        f"{all(parameter.grad is None for parameter in target_network.parameters())}"
    )


if __name__ == "__main__":
    run_demo()
