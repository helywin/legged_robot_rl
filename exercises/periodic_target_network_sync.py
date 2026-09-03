#!/usr/bin/env python3
"""第 56 课编程练习：按固定在线更新次数同步目标网络。

为什么有这道题
================

在线网络每次训练都会改参数，目标网络则要在若干次更新内保持固定，
然后再一次性复制在线参数。本题只学习“什么时候复制”和“怎样复制”。

本题使用只有一个权重的小网络。为了隔离同步规则，已提供的演示函数
每次直接让在线权重增加 `0.5`。这不是 DQN 训练，只是用可预测的参数变化
观察目标网络的同步时机。

固定过程
========

两个网络开始时权重都是 `1.0`，同步间隔为 3：

    更新次数    在线权重    是否同步    同步后目标权重
        1           1.5          False          1.0
        2           2.0          False          1.0
        3           2.5          True           2.5
        4           3.0          False          2.5
        5           3.5          False          2.5
        6           4.0          True           4.0

Python 取余数接口
================

`%` 返回除法的余数：

    1 % 3 == 1
    2 % 3 == 2
    3 % 3 == 0
    4 % 3 == 1
    5 % 3 == 2
    6 % 3 == 0

所以“当更新次整除同步间隔时同步”的条件是：

    update_number % sync_interval == 0

PyTorch 参数复制接口
==================

完整可运行接口是：

    online_state = online_network.state_dict()
    target_network.load_state_dict(online_state)

- `state_dict()` 返回“参数名 -> 参数值”的映射；
- 本题唯一的参数名是 `linear.weight`；
- `load_state_dict(...)` 把同名数值复制进目标网络已有参数；
- 它复制数值，不会让两个网络共用同一个参数对象。

你的任务
========

只修改 `maybe_sync_target_network()` 中的 TODO：

1. `sync_interval < 1` 时抛出 `ValueError`；
2. 未到同步次数时，不复制参数并返回 `False`；
3. 到达同步次数时，使用上面给出的 PyTorch 接口复制参数；
4. 复制后返回 `True`。

不要把 `target_network` 变量直接赋值为 `online_network`，不要创建 optimizer，
不要调用 `backward()`，不要修改演示循环或检查器。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.periodic_target_network_sync

成功条件
========

程序应显示 6 次权重变化，并且最后显示：

    sync_flags=[False, False, True, False, False, True] PASS
    target_values=[1.0, 1.0, 2.5, 2.5, 2.5, 4.0] PASS
    target_parameter_object_preserved=True PASS
    copied_values_are_not_shared=True PASS
    invalid_interval_rejected=True PASS

最后显示：

    练习通过：目标网络只在固定间隔复制在线参数

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

本题的在线参数变化是受控演示，不是 loss、`backward()` 和 optimizer 产生的真实
DQN 更新。没有完整 episode、CartPole 冒烟训练、检查点、独立评估、Isaac Lab
或真机验证。
"""

from __future__ import annotations

import torch
from torch import nn


class OneWeightNetwork(nn.Module):
    """只保存一个权重，便于直接观察参数复制。"""

    def __init__(self, initial_weight: float) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(initial_weight)


def read_weight(network: OneWeightNetwork) -> float:
    """读取唯一权重作为 Python 浮点数。"""
    return network.linear.weight.item()


def apply_known_online_change(
    online_network: OneWeightNetwork, change: float
) -> None:
    """受控地改变在线权重，只用于隔离观察同步时机。"""
    with torch.no_grad():
        online_network.linear.weight.add_(change)


def maybe_sync_target_network(
    online_network: nn.Module,
    target_network: nn.Module,
    update_number: int,
    sync_interval: int,
) -> bool:
    """到达固定更新间隔时，复制在线参数到同结构目标网络。"""
    # 1：拒绝小于 1 的 sync_interval。
    if sync_interval < 1:
        raise ValueError("sync_interval不能小于1")

    # 2：用 update_number % sync_interval 的余数判断是否到期。
    need_sync = update_number % sync_interval == 0

    # 3：到期时调用 state_dict() 和 load_state_dict(...)。
    if need_sync:
        model_dict = online_network.state_dict()
        target_network.load_state_dict(model_dict)

    # 4：未同步返回 False，已同步返回 True。
    return need_sync


def run_six_updates() -> tuple[
    list[float],
    list[float],
    list[bool],
    bool,
]:
    """运行 6 次受控参数变化，记录两个网络和同步时机。"""
    online_network = OneWeightNetwork(initial_weight=1.0)
    target_network = OneWeightNetwork(initial_weight=1.0)
    original_target_parameter = target_network.linear.weight
    online_values: list[float] = []
    target_values: list[float] = []
    sync_flags: list[bool] = []

    for update_number in range(1, 7):
        apply_known_online_change(online_network, change=0.5)
        synced = maybe_sync_target_network(
            online_network=online_network,
            target_network=target_network,
            update_number=update_number,
            sync_interval=3,
        )
        online_value = read_weight(online_network)
        target_value = read_weight(target_network)
        online_values.append(online_value)
        target_values.append(target_value)
        sync_flags.append(synced)
        print(
            f"update={update_number} "
            f"online={online_value:.1f} "
            f"target={target_value:.1f} "
            f"synced={synced}"
        )

    target_parameter_preserved = (
        target_network.linear.weight is original_target_parameter
    )
    return (
        online_values,
        target_values,
        sync_flags,
        target_parameter_preserved,
    )


def check_exercise() -> bool | None:
    """检查同步时机、数值复制、参数身份和非法间隔。"""
    try:
        online_values, target_values, sync_flags, parameter_preserved = (
            run_six_updates()
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    expected_online = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    expected_target = [1.0, 1.0, 2.5, 2.5, 2.5, 4.0]
    expected_flags = [False, False, True, False, False, True]

    online_values_ok = online_values == expected_online
    target_values_ok = target_values == expected_target
    sync_flags_ok = sync_flags == expected_flags

    # 第 4 次时在线权重继续变为 3.0，目标权重仍为 2.5。
    # 这证明第 3 次同步是复制数值，不是共享同一个参数。
    values_not_shared = (
        online_values_ok
        and target_values_ok
        and online_values[3] == 3.0
        and target_values[3] == 2.5
    )

    invalid_interval_rejected = False
    try:
        maybe_sync_target_network(
            online_network=OneWeightNetwork(1.0),
            target_network=OneWeightNetwork(1.0),
            update_number=1,
            sync_interval=0,
        )
    except ValueError:
        invalid_interval_rejected = True

    checks = (
        ("online_values_follow_known_changes=True", online_values_ok),
        (f"sync_flags={expected_flags}", sync_flags_ok),
        (f"target_values={expected_target}", target_values_ok),
        ("target_parameter_object_preserved=True", parameter_preserved),
        ("copied_values_are_not_shared=True", values_not_shared),
        ("invalid_interval_rejected=True", invalid_interval_rejected),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：目标网络只在固定间隔复制在线参数")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "maybe_sync_target_network() 中的 TODO"
        )


if __name__ == "__main__":
    main()
