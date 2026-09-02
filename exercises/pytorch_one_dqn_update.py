#!/usr/bin/env python3
"""第 49 课编程练习：把一条经验拼成一次完整 PyTorch DQN 更新。

为什么有这道题
============

前几课已经分别完成在线网络多动作预测、实际动作索引、目标网络 target、梯度清零、
反向计算和 optimizer 更新。本题不增加新算法部件，只把它们按正确职责连接起来。

本题只使用一条固定经验完成一次更新，不从回放缓冲区抽样，也不训练 CartPole。

场景
====

一条固定经验：

    observation = [0.2, -0.1]
    executed_action = 1
    reward = 0.2
    next_observation = [0.4, 0.2]
    terminated = False
    discount_factor = 0.9

更新前在线网络输出：

    q_values = [0.10, -0.05]
    selected_q = -0.05

目标网络得到：

    target_q = 1.01

平方损失：

    loss = (-0.05 - 1.01) ** 2 = 1.1236

已有接口
========

`one_dqn_update(...)` 接收在线网络、目标网络、只绑定在线参数的 optimizer 和一条
经验，必须返回更新前的：

    q_values, selected_q, target_q, loss

目标值计算函数 `calculate_dqn_target()` 已经提供，可以直接调用。

你的任务
========

只修改 `one_dqn_update()` 中的 TODO：

1. 清空在线 optimizer 的旧梯度；
2. 用在线网络预测旧观察的全部 Q 值；
3. 按 executed_action 取 selected_q；
4. 调用 calculate_dqn_target() 得到无梯度 target_q；
5. 计算 selected_q 与 target_q 的平方损失；
6. 调用反向计算；
7. 让 optimizer 更新在线网络；
8. 返回更新前的 q_values、selected_q、target_q 和 loss。

不要把 optimizer 绑定到目标网络，不要同步或修改目标网络，不要使用 `argmax()`
代替 executed_action，不要修改 `check_exercise()` 或 `main()`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_one_dqn_update

成功条件
========

程序应显示：

    selected_q_before=-0.0500 PASS
    target_q=1.0100 PASS
    loss=1.1236 PASS
    selected_q_after=0.1726 PASS
    selected_q_moved_toward_target=True PASS
    unselected_online_row_unchanged=True PASS
    target_parameters_unchanged=True PASS
    target_gradients_none=True PASS

最后显示：

    练习通过：一条经验已完成一次 PyTorch DQN 在线更新

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

本题没有经验回放抽样、批量张量、隐藏层、目标网络定期同步、CartPole 环境训练、
检查点或独立评估。
"""

from __future__ import annotations

import torch

from examples.pytorch_dqn_target_q import calculate_dqn_target
from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


def one_dqn_update(
    online_network: TwoActionQModule,
    target_network: TwoActionQModule,
    optimizer: torch.optim.Optimizer,
    observation: torch.Tensor,
    executed_action: int,
    reward: torch.Tensor,
    next_observation: torch.Tensor,
    discount_factor: float,
    terminated: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用一条经验更新在线网络，并返回更新前的关键张量。"""
    # 只修改这里，拼接预测、动作索引、target、loss、backward 和 step。
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
    return (q_values, selected_q, target_q, loss)


def check_exercise() -> bool | None:
    """检查一次 DQN 更新的数据流和两个网络的职责。"""
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

    try:
        q_values, selected_q, target_q, loss = one_dqn_update(
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
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    with torch.no_grad():
        selected_q_after = online_network(observation)[1]

    q_values_ok = (
        isinstance(q_values, torch.Tensor)
        and q_values.shape == (2,)
        and torch.allclose(q_values, torch.tensor([0.1, -0.05]))
    )
    selected_ok = (
        isinstance(selected_q, torch.Tensor)
        and selected_q.ndim == 0
        and abs(selected_q.item() + 0.05) < 1e-6
    )
    target_ok = (
        isinstance(target_q, torch.Tensor)
        and target_q.ndim == 0
        and not target_q.requires_grad
        and abs(target_q.item() - 1.01) < 1e-6
    )
    loss_ok = (
        isinstance(loss, torch.Tensor)
        and loss.ndim == 0
        and abs(loss.item() - 1.1236) < 1e-6
    )
    after_ok = abs(selected_q_after.item() - 0.1726) < 1e-6
    moved_toward = (
        selected_ok
        and target_ok
        and abs(selected_q_after.item() - target_q.item())
        < abs(selected_q.item() - target_q.item())
    )
    unselected_unchanged = (
        torch.equal(online_before[0][0], online_network.q_values.weight[0])
        and online_before[1][0].item()
        == online_network.q_values.bias[0].item()
    )
    target_unchanged = all(
        torch.equal(old, current)
        for old, current in zip(target_before, target_network.parameters())
    )
    target_gradients_none = all(
        parameter.grad is None for parameter in target_network.parameters()
    )

    checks = (
        ("q_values_before=[0.1, -0.05]", q_values_ok),
        ("selected_q_before=-0.0500", selected_ok),
        ("target_q=1.0100", target_ok),
        ("loss=1.1236", loss_ok),
        ("selected_q_after=0.1726", after_ok),
        ("selected_q_moved_toward_target=True", moved_toward),
        ("unselected_online_row_unchanged=True", unselected_unchanged),
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
        print("练习通过：一条经验已完成一次 PyTorch DQN 在线更新")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "one_dqn_update() 中的 TODO"
        )


if __name__ == "__main__":
    main()
