#!/usr/bin/env python3
"""第 48 课编程练习：用奖励和下一观察计算 DQN 目标 Q 值。

为什么有这道题
============

上一课的 target 是人工给定的。真实 DQN 会从经验中的奖励、下一观察和 terminated
计算 target。下一观察的 Q 值由暂时固定的目标网络提供，而且 target 不应连接到
目标网络的梯度。

本题只计算 target，不使用在线网络、不计算 selected_q 损失，也不更新参数。

场景
====

固定下一观察经过目标网络得到：

    next_q_values = [0.9, -0.1]
    best_next_q = 0.9
    reward = 0.2
    discount_factor = 0.9

如果任务尚未真正结束：

    target = reward + discount_factor * best_next_q
           = 0.2 + 0.9 * 0.9
           = 1.01

如果 `terminated=True`，任务状态已经结束，没有未来动作：

    target = reward = 0.2

已有接口
========

`calculate_target(target_network, next_observation, reward,
discount_factor, terminated)` 返回一个标量 target 张量。

你的任务
========

只修改 `calculate_target()` 中的 TODO：

1. 整个 target 计算不记录梯度；
2. 如果 terminated 为 True，直接返回只含 reward 的目标；
3. 否则让 target_network 预测下一观察的全部 Q 值；
4. 取其中最大 Q 值；
5. 返回 reward 加折扣后的最大未来 Q 值。

不要调用 `backward()` 或 optimizer，不要使用在线网络，不要修改目标网络参数，
不要修改 `check_exercise()` 或 `main()`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_dqn_target_q

成功条件
========

程序应显示：

    next_q_values=[0.9, -0.1] PASS
    continuing_target=1.010 PASS
    terminal_target=0.200 PASS
    target_requires_grad=False PASS
    target_network_gradients_none=True PASS
    target_parameters_unchanged=True PASS

最后显示：

    练习通过：奖励与下一观察已组成无梯度的 DQN 目标值

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

本题没有把 target 与在线网络的 selected_q 组成损失，也没有 optimizer 更新、经验
回放抽样、CartPole 环境训练或独立评估。
"""

from __future__ import annotations

import torch

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


def calculate_target(
    target_network: TwoActionQModule,
    next_observation: torch.Tensor,
    reward: torch.Tensor,
    discount_factor: float,
    terminated: bool,
) -> torch.Tensor:
    """返回不连接梯度的标量 DQN target。"""
    if terminated:
        return reward
    else:
        with torch.no_grad():
            q_values = target_network(next_observation)
            best_next_q = max(q_values)
            target = reward + discount_factor * best_next_q
            return target


def check_exercise() -> bool | None:
    """检查两类 target、梯度隔离和参数不变性。"""
    target_network = TwoActionQModule()
    set_demo_parameters(target_network)
    next_observation = torch.tensor([0.4, 0.2], dtype=torch.float32)
    reward = torch.tensor(0.2, dtype=torch.float32)
    before = [
        parameter.detach().clone()
        for parameter in target_network.parameters()
    ]

    with torch.no_grad():
        next_q_values = target_network(next_observation)

    try:
        continuing_target = calculate_target(
            target_network,
            next_observation,
            reward,
            discount_factor=0.9,
            terminated=False,
        )
        terminal_target = calculate_target(
            target_network,
            next_observation,
            reward,
            discount_factor=0.9,
            terminated=True,
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    next_values_ok = torch.allclose(
        next_q_values, torch.tensor([0.9, -0.1])
    )
    continuing_ok = (
        isinstance(continuing_target, torch.Tensor)
        and continuing_target.ndim == 0
        and abs(continuing_target.item() - 1.01) < 1e-6
    )
    terminal_ok = (
        isinstance(terminal_target, torch.Tensor)
        and terminal_target.ndim == 0
        and abs(terminal_target.item() - 0.2) < 1e-6
    )
    targets_detached = (
        continuing_ok
        and terminal_ok
        and not continuing_target.requires_grad
        and not terminal_target.requires_grad
    )
    gradients_none = all(
        parameter.grad is None for parameter in target_network.parameters()
    )
    unchanged = all(
        torch.equal(old, current)
        for old, current in zip(before, target_network.parameters())
    )

    checks = (
        ("next_q_values=[0.9, -0.1]", next_values_ok),
        ("continuing_target=1.010", continuing_ok),
        ("terminal_target=0.200", terminal_ok),
        ("target_requires_grad=False", targets_detached),
        ("target_network_gradients_none=True", gradients_none),
        ("target_parameters_unchanged=True", unchanged),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：奖励与下一观察已组成无梯度的 DQN 目标值")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "calculate_target() 中的 TODO"
        )


if __name__ == "__main__":
    main()
