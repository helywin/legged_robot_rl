#!/usr/bin/env python3
"""第 57 课编程练习：连续执行回放更新并定期同步目标网络。

为什么有这道题
================

第 55 课已经完成一次真实回放更新，第 56 课已经完成“到期才复制目标参数”。
训练不会只做一次更新，所以本题要把两件事按固定顺序放进同一个循环：

    抽样并更新在线网络一次
    -> 记录这次 loss
    -> 判断本次更新后是否同步目标网络
    -> 进入下一次更新

本题还不运行环境 episode，只运行回放训练部分。

固定输入与预期
============

- 回放缓冲区：第 55 课的三条固定经验；
- 随机数对象：循环外创建的同一个 `random.Random(7)`；
- 每次抽样：2 条经验；
- 折扣因子：0.9；
- 在线更新次数：6；
- 目标同步间隔：3 次在线更新。

预期轨迹：

    update=1 loss=0.641800 synced=False target_matches_online=False
    update=2 loss=0.511266 synced=False target_matches_online=False
    update=3 loss=0.379486 synced=True  target_matches_online=True
    update=4 loss=0.378828 synced=False target_matches_online=False
    update=5 loss=0.261590 synced=False target_matches_online=False
    update=6 loss=0.209546 synced=True  target_matches_online=True

每次 loss 是本次 `optimizer.step()` 之前用旧在线参数计算出的损失，也是本次
`backward()` 使用的那个 loss。它不是更新参数后重新前向计算的 loss。

已知接口 1：一次回放更新
==========================

完整调用形式是：

    result = train_from_replay(
        replay_buffer=replay_buffer,
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        random_generator=random_generator,
        batch_size=batch_size,
        discount_factor=discount_factor,
    )

返回顺序是：

    (
        online_q_values,
        selected_q_values,
        target_q_values,
        per_item_losses,
        loss,
    )

本题只需要第 5 项 `loss`。可以先保存整个 `result`，再使用：

    loss = result[4]

已知接口 2：记录 loss 数字
==========================

`loss` 是只含一个数字的标量张量。转成普通 Python `float` 用：

    loss_value = loss.item()

`.item()` 只读出数字用于记录，不会再执行参数更新。本次的 `backward()` 和
`optimizer.step()` 已经在 `train_from_replay()` 内部完成。

已知接口 3：判断并执行同步
============================

完整调用形式是：

    synced = maybe_sync_target_network(
        online_network=online_network,
        target_network=target_network,
        update_number=update_number,
        sync_interval=sync_interval,
    )

它未到期时返回 `False`，到期复制参数后返回 `True`。

Python 循环与随机数对象
====================

更新编号从 1 到 `update_count` 时，循环形式是：

    for update_number in range(1, update_count + 1):
        ...

`range` 的结束数不包含在结果里，所以上界要写 `update_count + 1`。当
`update_count=6` 时，实际编号是 `1,2,3,4,5,6`。

`random_generator` 必须是循环外创建并传入的同一个对象。每抽样一次，它的内部状态
就向后走一段，下一轮才会得到后续随机结果。不要在循环内反复创建
`random.Random(7)`，否则每轮都会回到同一个随机序列起点。

你的任务
========

只修改 `run_repeated_replay_updates()` 中的 TODO：

1. 拒绝小于 1 的 `update_count`；
2. 创建三个空列表，分别记录 loss、同步标记和两网参数是否相同；
3. 按 `1..update_count` 循环，每轮先完成一次回放更新；
4. 用 `result[4].item()` 记录本次 loss；
5. 在更新后检查并执行目标网络同步；
6. 调用已提供的 `parameters_match(...)` 记录同步后两网参数是否相同；
7. 按函数返回类型中的顺序返回三个列表。

不要重新创建随机数对象、手写 DQN 公式、多调或少调 `train_from_replay()`、
在更新前同步、修改回放缓冲区或修改检查器。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.repeated_replay_updates

成功条件
========

程序应显示 6 次更新，并最后确认：

    six_losses_match_reference=True PASS
    sync_flags=[False, False, True, False, False, True] PASS
    target_matches=[False, False, True, False, False, True] PASS
    replay_buffer_unchanged=True PASS
    online_parameters_changed=True PASS
    final_target_matches_online=True PASS
    invalid_update_count_rejected=True PASS

最后显示：

    练习通过：6 次回放更新已与每 3 次目标同步接成循环

TODO 未完成时只显示友好提示，不会输出异常堆栈或破坏全量测试。

当前没有验证
============

本题重复使用固定的三条经验，没有环境 `step()` 持续产生新经验、完整 episode、
探索策略、CartPole 冒烟训练、检查点、独立评估、Isaac Lab 或真机验证。
"""

from __future__ import annotations

import random

import torch
from torch import nn

from examples.pytorch_two_action_q_values import TwoActionQModule
from exercises.periodic_target_network_sync import (
    maybe_sync_target_network,
)
from exercises.replay_sample_batch_update import (
    VectorReplayBuffer,
    make_replay_buffer,
    make_training_objects,
    train_from_replay,
)


def parameters_match(
    online_network: nn.Module, target_network: nn.Module
) -> bool:
    """只读比较两个同结构网络的所有参数值。"""
    return all(
        torch.equal(online_parameter, target_parameter)
        for online_parameter, target_parameter in zip(
            online_network.parameters(), target_network.parameters()
        )
    )


def run_repeated_replay_updates(
    replay_buffer: VectorReplayBuffer,
    online_network: TwoActionQModule,
    target_network: TwoActionQModule,
    optimizer: torch.optim.Optimizer,
    random_generator: random.Random,
    update_count: int,
    batch_size: int,
    discount_factor: float,
    sync_interval: int,
) -> tuple[list[float], list[bool], list[bool]]:
    """连续执行回放更新，每次更新后检查目标同步。"""
    # 1：拒绝小于 1 的 update_count。
    if update_count < 1:
        raise ValueError("update_count小于1")

    # 2：创建 losses、sync_flags、target_matches 三个空列表。
    losses = []
    sync_flags = []
    target_matches = []

    # 3：用 range(1, update_count + 1) 产生更新编号。
    indexes = range(1, update_count + 1)

    # 4：每轮先调用 train_from_replay(...)，再从 result[4]
    # 取出 loss，使用 .item() 记录 Python 数字。
    for index in indexes:
        result = train_from_replay(replay_buffer, online_network, target_network, optimizer, random_generator, batch_size, discount_factor)
        losses.append(result[4].item())

    # 5：在本次在线更新后调用 maybe_sync_target_network(...)，
    # 把返回布尔值记录到 sync_flags。
        sync_flag = maybe_sync_target_network(online_network, target_network, index, sync_interval)
        sync_flags.append(sync_flag)
    # 6：调用 parameters_match(...) 记录同步后的参数关系。
        target_match = parameters_match(online_network, target_network)
        target_matches.append(target_match)
    # 7：返回 losses、sync_flags、target_matches。
    return (losses, sync_flags, target_matches)


def check_exercise() -> bool | None:
    """检查六次 loss、同步时机、参数关系和输入边界。"""
    replay_buffer = make_replay_buffer()
    online_network, target_network, optimizer = make_training_objects()
    buffer_before = replay_buffer.snapshot()
    online_before = [
        parameter.detach().clone()
        for parameter in online_network.parameters()
    ]

    try:
        losses, sync_flags, target_matches = run_repeated_replay_updates(
            replay_buffer=replay_buffer,
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            random_generator=random.Random(7),
            update_count=6,
            batch_size=2,
            discount_factor=0.9,
            sync_interval=3,
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    expected_losses = [
        0.641800,
        0.511266,
        0.379486,
        0.378828,
        0.261590,
        0.209546,
    ]
    expected_flags = [False, False, True, False, False, True]
    losses_ok = len(losses) == 6 and all(
        abs(actual - expected) < 1e-5
        for actual, expected in zip(losses, expected_losses)
    )
    sync_flags_ok = sync_flags == expected_flags
    target_matches_ok = target_matches == expected_flags
    buffer_unchanged = replay_buffer.snapshot() == buffer_before
    online_changed = any(
        not torch.equal(old, current)
        for old, current in zip(
            online_before, online_network.parameters()
        )
    )
    final_target_matches = parameters_match(
        online_network, target_network
    )

    invalid_update_count_rejected = False
    invalid_online, invalid_target, invalid_optimizer = (
        make_training_objects()
    )
    try:
        run_repeated_replay_updates(
            replay_buffer=make_replay_buffer(),
            online_network=invalid_online,
            target_network=invalid_target,
            optimizer=invalid_optimizer,
            random_generator=random.Random(7),
            update_count=0,
            batch_size=2,
            discount_factor=0.9,
            sync_interval=3,
        )
    except ValueError:
        invalid_update_count_rejected = True

    for index, (loss, synced, matches) in enumerate(
        zip(losses, sync_flags, target_matches), start=1
    ):
        print(
            f"update={index} "
            f"loss={loss:.6f} "
            f"synced={synced} "
            f"target_matches_online={matches}"
        )

    checks = (
        ("six_losses_match_reference=True", losses_ok),
        (f"sync_flags={expected_flags}", sync_flags_ok),
        (f"target_matches={expected_flags}", target_matches_ok),
        ("replay_buffer_unchanged=True", buffer_unchanged),
        ("online_parameters_changed=True", online_changed),
        ("final_target_matches_online=True", final_target_matches),
        ("invalid_update_count_rejected=True", invalid_update_count_rejected),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：6 次回放更新已与每 3 次目标同步接成循环")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "run_repeated_replay_updates() 中的 TODO"
        )


if __name__ == "__main__":
    main()
