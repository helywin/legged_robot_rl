#!/usr/bin/env python3
"""第 53 课编程练习：把在线支路和目标支路合成一次完整批量 DQN 更新。

为什么有这道题
============

第 51 课完成了在线预测、批量平均损失和参数更新；第 52 课完成了目标网络、终止
掩码和无梯度 target。本题不增加新公式，只把同一批经验的两条支路在 loss 处汇合，
并验证 optimizer 只修改在线网络。

固定批量
========

    observations = [[0.2, -0.1], [-0.3, 0.4]]
    executed_actions = [1, 0]
    rewards = [0.2, 1.0]
    next_observations = [[0.4, 0.2], [-0.2, 0.3]]
    terminated = [False, True]
    discount_factor = 0.9

更新前关键结果：

    selected_q_values = [-0.05, 0.60]
    target_q_values = [1.01, 1.00]
    per_item_losses = [1.1236, 0.1600]
    loss = 0.6418

学习率为 0.1，更新后：

    selected_q_after = [0.0613, 0.65]
    loss_after = 0.5113

你的任务
========

只修改 `full_batch_dqn_update()` 中的 TODO：

1. 清空在线 optimizer 的旧梯度；
2. 旧观察整批进入 online_network，并按实际动作逐行取 selected Q；
3. 在 `torch.no_grad()` 中，让下一观察整批进入 target_network；
4. 逐行取目标网络最大下一 Q 值，使用终止掩码计算 target Q；
5. 计算逐条平方损失，再取平均得到标量 loss；
6. 调用反向计算，再让 optimizer 更新在线参数一次；
7. 返回 online_q_values、selected_q_values、target_q_values、per_item_losses 和 loss。

动作索引和压缩方式沿用第 50 课；终止掩码沿用第 52 课。不要逐条调用网络或
optimizer，不要让 target 保留梯度，不要修改或同步目标网络，不要修改检查器或 main。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_full_batch_dqn_update

成功条件
========

程序应显示：

    selected_q_before=[-0.05, 0.6] PASS
    target_q_values=[1.01, 1.0] PASS
    per_item_losses=[1.1236, 0.16] PASS
    mean_loss=0.6418 PASS
    selected_q_after=[0.0613, 0.65] PASS
    both_predictions_moved_toward_targets=True PASS
    online_parameters_changed=True PASS
    target_parameters_unchanged=True PASS
    target_gradients_none=True PASS

最后显示：

    练习通过：在线支路和目标支路已合成一次完整批量 DQN 更新

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏仓库全量测试。

当前没有验证
============

本题使用直接构造的批量张量，没有经验回放随机抽样、目标网络定期同步、多层网络、
完整 CartPole 训练、检查点或独立评估。
"""

from __future__ import annotations

import torch

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


def full_batch_dqn_update(
    online_network: TwoActionQModule,
    target_network: TwoActionQModule,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    executed_actions: torch.Tensor,
    rewards: torch.Tensor,
    next_observations: torch.Tensor,
    terminated: torch.Tensor,
    discount_factor: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """使用一批经验更新在线网络一次，并返回更新前关键张量。"""
    # 只修改这里，拼接在线批量支路、无梯度目标支路和一次更新。
    optimizer.zero_grad()
    q_values = online_network(observations)
    indexes = executed_actions.unsqueeze(1)
    selected_q_values = q_values.gather(dim=1, index=indexes).squeeze(1)
    target_q_values : torch.Tensor
    with torch.no_grad():
        next_q_values = target_network(next_observations)
        best_next_q = next_q_values.max(dim=1).values
        future_mask = (~terminated).float()
        target_q_values = rewards + best_next_q * discount_factor * future_mask
    loss = (target_q_values - selected_q_values) ** 2
    mean_loss = loss.mean()
    mean_loss.backward()
    optimizer.step()
    return (q_values, selected_q_values, target_q_values, loss , mean_loss)

def check_exercise() -> bool | None:
    """检查两条支路、批量损失和两个网络的职责边界。"""
    online_network = TwoActionQModule()
    target_network = TwoActionQModule()
    set_demo_parameters(online_network)
    set_demo_parameters(target_network)
    optimizer = torch.optim.SGD(online_network.parameters(), lr=0.1)
    observations = torch.tensor(
        [[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float32
    )
    executed_actions = torch.tensor([1, 0], dtype=torch.long)
    rewards = torch.tensor([0.2, 1.0], dtype=torch.float32)
    next_observations = torch.tensor(
        [[0.4, 0.2], [-0.2, 0.3]], dtype=torch.float32
    )
    terminated = torch.tensor([False, True], dtype=torch.bool)
    online_before = [
        parameter.detach().clone()
        for parameter in online_network.parameters()
    ]
    target_before = [
        parameter.detach().clone()
        for parameter in target_network.parameters()
    ]

    try:
        online_q, selected_q, targets, per_item_losses, loss = (
            full_batch_dqn_update(
                online_network,
                target_network,
                optimizer,
                observations,
                executed_actions,
                rewards,
                next_observations,
                terminated,
                discount_factor=0.9,
            )
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    with torch.no_grad():
        online_after = online_network(observations)
        selected_after = online_after.gather(
            1, executed_actions.unsqueeze(1)
        ).squeeze(1)
        loss_after = ((selected_after - targets) ** 2).mean()

    online_q_ok = (
        isinstance(online_q, torch.Tensor)
        and online_q.shape == (2, 2)
        and torch.allclose(
            online_q, torch.tensor([[0.1, -0.05], [0.6, 0.7]])
        )
    )
    selected_ok = (
        isinstance(selected_q, torch.Tensor)
        and selected_q.shape == (2,)
        and torch.allclose(selected_q, torch.tensor([-0.05, 0.6]))
    )
    targets_ok = (
        isinstance(targets, torch.Tensor)
        and targets.shape == (2,)
        and not targets.requires_grad
        and torch.allclose(targets, torch.tensor([1.01, 1.0]))
    )
    per_item_ok = (
        isinstance(per_item_losses, torch.Tensor)
        and per_item_losses.shape == (2,)
        and torch.allclose(
            per_item_losses, torch.tensor([1.1236, 0.16])
        )
    )
    loss_ok = (
        isinstance(loss, torch.Tensor)
        and loss.ndim == 0
        and abs(loss.item() - 0.6418) < 1e-6
    )
    selected_after_ok = torch.allclose(
        selected_after, torch.tensor([0.0613, 0.65]), atol=1e-6
    )
    moved_toward = selected_ok and targets_ok and all(
        abs(after.item() - target.item())
        < abs(before.item() - target.item())
        for before, after, target in zip(
            selected_q, selected_after, targets
        )
    )
    loss_decreased = loss_ok and loss_after.item() < loss.item()
    online_changed = any(
        not torch.equal(old, current)
        for old, current in zip(
            online_before, online_network.parameters()
        )
    )
    target_unchanged = all(
        torch.equal(old, current)
        for old, current in zip(
            target_before, target_network.parameters()
        )
    )
    target_gradients_none = all(
        parameter.grad is None
        for parameter in target_network.parameters()
    )

    checks = (
        ("online_q_values_shape=(2, 2)", online_q_ok),
        ("selected_q_before=[-0.05, 0.6]", selected_ok),
        ("target_q_values=[1.01, 1.0]", targets_ok),
        ("per_item_losses=[1.1236, 0.16]", per_item_ok),
        ("mean_loss=0.6418", loss_ok),
        ("selected_q_after=[0.0613, 0.65]", selected_after_ok),
        ("both_predictions_moved_toward_targets=True", moved_toward),
        ("loss_decreased=True", loss_decreased),
        ("online_parameters_changed=True", online_changed),
        ("target_parameters_unchanged=True", target_unchanged),
        ("target_gradients_none=True", target_gradients_none),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print(
            "练习通过：在线支路和目标支路已合成一次完整批量 DQN 更新"
        )
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "full_batch_dqn_update() 中的 TODO"
        )


if __name__ == "__main__":
    main()
