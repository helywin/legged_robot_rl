#!/usr/bin/env python3
"""第 52 课编程练习：为一批经验分别处理终止与未来价值。

为什么有这道题
============

一批经验中的 `terminated` 可能不同。目标网络可以整批计算下一观察，但真正终止的
经验不能保留未来价值。本题使用逐行终止掩码，让未终止行保留未来项、终止行把
未来项乘成零。

固定数据
========

    next_observations = [
        [ 0.4, 0.2],
        [-0.2, 0.3],
    ]
    rewards = [0.2, 1.0]
    terminated = [False, True]
    discount_factor = 0.9

目标网络输出：

    next_q_values = [
        [0.90, -0.10],
        [0.50,  0.55],
    ]
    best_next_q = [0.90, 0.55]

未来掩码与 target：

    future_mask = [1.0, 0.0]
    target_q_values = [1.01, 1.00]

你的任务
========

只修改 `calculate_batch_dqn_targets()` 中的 TODO：

1. 将后续所有计算放入 `torch.no_grad()`；
2. 让 next_observations 整批进入 target_network；
3. 使用张量 `max(dim=1)` 的 `.values` 取得每行最大 Q 值；
4. 对 terminated 使用 `~` 取反，再转换为与 rewards 相同的浮点 dtype，得到未来掩码；
5. 计算 `rewards + discount_factor * best_next_q * future_mask`；
6. 返回 next_q_values、best_next_q、future_mask 和 target_q_values。

不要逐条调用目标网络，不要使用 Python `if` 处理整批，不要把 `truncated` 混入
terminated，不要调用 backward 或 optimizer，也不要修改检查器或 main。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_batch_dqn_targets

成功条件
========

程序应显示：

    next_q_values=[[0.9, -0.1], [0.5, 0.55]] PASS
    best_next_q=[0.9, 0.55] PASS
    future_mask=[1.0, 0.0] PASS
    target_q_values=[1.01, 1.0] PASS
    target_requires_grad=False PASS
    target_gradients_none=True PASS
    target_parameters_unchanged=True PASS

最后显示：

    练习通过：一批 DQN target 已分别处理终止与未来价值

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏仓库全量测试。

当前没有验证
============

本题没有在线网络预测、loss、backward、optimizer、经验回放随机抽样、完整 CartPole
训练、检查点或独立评估。
"""

from __future__ import annotations

import torch

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


def calculate_batch_dqn_targets(
    target_network: TwoActionQModule,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    discount_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回批量下一 Q 值、最大值、未来掩码和无梯度 target。"""
    # 只修改这里，完成无梯度批量目标网络前向与终止掩码。
    with torch.no_grad():
        next_q_values = target_network(next_observations)
        best_next_q = next_q_values.max(dim=1).values
        future_mask = (~terminated).float()
        target_q_values = rewards + discount_factor * best_next_q * future_mask
        return (next_q_values, best_next_q, future_mask, target_q_values)

def check_exercise() -> bool | None:
    """检查批量 target 数值、终止语义和目标网络隔离。"""
    target_network = TwoActionQModule()
    set_demo_parameters(target_network)
    next_observations = torch.tensor(
        [[0.4, 0.2], [-0.2, 0.3]], dtype=torch.float32
    )
    rewards = torch.tensor([0.2, 1.0], dtype=torch.float32)
    terminated = torch.tensor([False, True], dtype=torch.bool)
    target_before = [
        parameter.detach().clone()
        for parameter in target_network.parameters()
    ]

    try:
        next_q, best_next, future_mask, targets = (
            calculate_batch_dqn_targets(
                target_network,
                next_observations,
                rewards,
                terminated,
                discount_factor=0.9,
            )
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    next_q_ok = (
        isinstance(next_q, torch.Tensor)
        and next_q.shape == (2, 2)
        and torch.allclose(
            next_q, torch.tensor([[0.9, -0.1], [0.5, 0.55]])
        )
    )
    best_next_ok = (
        isinstance(best_next, torch.Tensor)
        and best_next.shape == (2,)
        and torch.allclose(best_next, torch.tensor([0.9, 0.55]))
    )
    mask_ok = (
        isinstance(future_mask, torch.Tensor)
        and future_mask.shape == (2,)
        and future_mask.dtype == rewards.dtype
        and torch.equal(future_mask, torch.tensor([1.0, 0.0]))
    )
    targets_ok = (
        isinstance(targets, torch.Tensor)
        and targets.shape == (2,)
        and torch.allclose(targets, torch.tensor([1.01, 1.0]))
    )
    no_grad = all(
        isinstance(value, torch.Tensor) and not value.requires_grad
        for value in (next_q, best_next, future_mask, targets)
    )
    target_gradients_none = all(
        parameter.grad is None for parameter in target_network.parameters()
    )
    target_parameters_unchanged = all(
        torch.equal(old, current)
        for old, current in zip(target_before, target_network.parameters())
    )

    checks = (
        ("next_q_values=[[0.9, -0.1], [0.5, 0.55]]", next_q_ok),
        ("best_next_q=[0.9, 0.55]", best_next_ok),
        ("future_mask=[1.0, 0.0]", mask_ok),
        ("target_q_values=[1.01, 1.0]", targets_ok),
        ("target_requires_grad=False", no_grad),
        ("target_gradients_none=True", target_gradients_none),
        ("target_parameters_unchanged=True", target_parameters_unchanged),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：一批 DQN target 已分别处理终止与未来价值")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "calculate_batch_dqn_targets() 中的 TODO"
        )


if __name__ == "__main__":
    main()
