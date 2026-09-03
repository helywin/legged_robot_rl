#!/usr/bin/env python3
"""第 58 课编程练习：环境先加入新经验，达到预填充数量后再训练。

为什么有这道题
================

第 57 课在固定的三条经验上重复更新，便于验证循环，但它不会产生新信息。
真实在线 DQN 会让环境继续产生经验。开始阶段先只收集，回放缓冲区达到
`warmup_size` 后，才允许第一次参数更新。

“预填充”不是网络预训练。这个阶段环境仍在交互，只是 optimizer 暂时不运行。

固定时间线
==========

本题提供 6 条按环境时间顺序到来的经验，设置：

    warmup_size = 4
    batch_size = 2
    sync_interval = 3

预期时间线：

    env_step=1 buffer_size=1 trained=False update_number=0
    env_step=2 buffer_size=2 trained=False update_number=0
    env_step=3 buffer_size=3 trained=False update_number=0
    env_step=4 buffer_size=4 trained=True  update_number=1 loss=0.569000 synced=False
    env_step=5 buffer_size=5 trained=True  update_number=2 loss=0.981785 synced=False
    env_step=6 buffer_size=6 trained=True  update_number=3 loss=0.481624 synced=True

第 1～3 个环境 step 不是什么都没做：它们分别产生并保存了一条新经验。
只是因为数量还没达到 4，所以在线参数更新数仍然是 0。

已知 Python 接口
================

读取缓冲区当前数量：

    buffer_size = len(replay_buffer)

`len(...)` 会调用对象的 `__len__()`，本题的缓冲区已经实现。

判断是否达到预填充数量：

    if len(replay_buffer) >= warmup_size:
        ...

只在进入这个 `if` 时，才增加在线更新计数：

    update_number += 1

这等价于：

    update_number = update_number + 1

已知训练接口
================

一次回放更新仍使用第 55 课的完整接口：

    result = train_from_replay(
        replay_buffer=replay_buffer,
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        random_generator=random_generator,
        batch_size=batch_size,
        discount_factor=discount_factor,
    )

本次 loss 仍使用：

    loss_value = result[4].item()

每次在线更新完成后，使用第 56 课的接口判断目标同步：

    synced = maybe_sync_target_network(
        online_network=online_network,
        target_network=target_network,
        update_number=update_number,
        sync_interval=sync_interval,
    )

你的任务
========

只修改 `collect_then_train()` 中的 TODO：

1. `warmup_size < batch_size` 时抛出 `ValueError`，且不能先改缓冲区；
2. 创建结果列表和初始为 0 的 `update_number`；
3. 逐条遍历 `transition_stream`，每次先调用 `replay_buffer.add(transition)`；
4. 只在缓冲区数量大于等于 `warmup_size` 时更新；
5. 每次真正更新时，记录当前环境 step、loss 和同步标记；
6. 按函数返回类型的顺序返回四项结果。

不要在预填充阶段调用训练函数，不要用环境 step 代替 `update_number`，不要
重建随机数对象，不要重写 DQN 公式，不要修改经验流或检查器。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.replay_warmup

成功条件
========

最后应确认：

    trained_at_env_steps=[4, 5, 6] PASS
    three_losses_match_reference=True PASS
    sync_flags=[False, False, True] PASS
    update_count=3 PASS
    final_buffer_size=6 PASS
    online_parameters_changed=True PASS
    final_target_matches_online=True PASS
    invalid_warmup_rejected_before_collection=True PASS

最后显示：

    练习通过：先收集 4 条经验，再开始回放更新

TODO 未完成时只显示友好提示，不会输出异常堆栈或破坏全量测试。

当前没有验证
============

本题使用预先写好的经验流模拟环境时间顺序，没有调用 Gymnasium `reset/step`，
没有探索策略、完整 episode、CartPole 冒烟训练、检查点、独立评估、Isaac Lab
或真机验证。
"""

from __future__ import annotations

import random

import torch

from examples.pytorch_two_action_q_values import TwoActionQModule
from exercises.periodic_target_network_sync import (
    maybe_sync_target_network,
)
from exercises.repeated_replay_updates import parameters_match
from exercises.replay_sample_batch_update import (
    VectorReplayBuffer,
    make_training_objects,
    train_from_replay,
)
from exercises.replay_samples_to_tensors import VectorTransition


def make_transition_stream() -> list[VectorTransition]:
    """返回按环境时间顺序到来的 6 条固定经验。"""
    return [
        VectorTransition(
            observation=(0.2, -0.1),
            action=1,
            reward=0.2,
            next_observation=(0.4, 0.2),
            terminated=False,
            truncated=False,
        ),
        VectorTransition(
            observation=(-0.3, 0.4),
            action=0,
            reward=1.0,
            next_observation=(-0.2, 0.3),
            terminated=True,
            truncated=False,
        ),
        VectorTransition(
            observation=(0.0, 0.0),
            action=0,
            reward=-0.2,
            next_observation=(0.1, 0.0),
            terminated=False,
            truncated=False,
        ),
        VectorTransition(
            observation=(0.5, -0.2),
            action=1,
            reward=0.3,
            next_observation=(0.6, -0.1),
            terminated=False,
            truncated=False,
        ),
        VectorTransition(
            observation=(-0.4, 0.1),
            action=0,
            reward=0.0,
            next_observation=(-0.3, 0.2),
            terminated=False,
            truncated=False,
        ),
        VectorTransition(
            observation=(0.1, 0.2),
            action=1,
            reward=0.5,
            next_observation=(0.2, 0.3),
            terminated=True,
            truncated=False,
        ),
    ]


def collect_then_train(
    transition_stream: list[VectorTransition],
    replay_buffer: VectorReplayBuffer,
    online_network: TwoActionQModule,
    target_network: TwoActionQModule,
    optimizer: torch.optim.Optimizer,
    random_generator: random.Random,
    warmup_size: int,
    batch_size: int,
    discount_factor: float,
    sync_interval: int,
) -> tuple[list[int], list[float], list[bool], int]:
    """按时间加入经验，达到预填充数量后才开始更新。"""
    # 1：在收集任何经验前，检查 warmup_size >= batch_size。
    if warmup_size < batch_size:
        raise ValueError("warmup_size过小")

    # 2：创建 trained_env_steps、losses、sync_flags 和
    # 初始为 0 的 update_number、env_step。
    trained_env_steps = []
    losses = []
    sync_flags = []
    update_number = 0
    env_step = 0

    # 3：逐条遍历 transition_stream。每轮先让 env_step 加 1，
    # 再调用 replay_buffer.add(transition)。
    for transition in transition_stream:
        env_step += 1
        replay_buffer.add(transition)

    # 4：只在 len(replay_buffer) >= warmup_size 时进入训练支路。
        if len(replay_buffer) >= warmup_size:

    # 5：训练支路中先让 update_number 加 1，再调用
    # train_from_replay(...)，记录 env_step 和 result[4].item()。
            update_number += 1
            result = train_from_replay(replay_buffer, online_network, target_network, optimizer, random_generator, batch_size, discount_factor)
            losses.append(result[4].item())
            trained_env_steps.append(env_step)
        
    # 6：在本次在线更新后调用 maybe_sync_target_network(...)，
    # 使用 update_number 而不是 env_step，记录返回布尔值。
            sync_flag = maybe_sync_target_network(online_network, target_network, update_number, sync_interval)
            sync_flags.append(sync_flag)
    
    # 7：返回 trained_env_steps、losses、sync_flags、update_number。
    return (trained_env_steps, losses, sync_flags, update_number)

def check_exercise() -> bool | None:
    """检查预填充时间线、真实更新结果和非法配置边界。"""
    replay_buffer = VectorReplayBuffer(capacity=6)
    online_network, target_network, optimizer = make_training_objects()
    online_before = [
        parameter.detach().clone()
        for parameter in online_network.parameters()
    ]

    try:
        trained_steps, losses, sync_flags, update_count = (
            collect_then_train(
                transition_stream=make_transition_stream(),
                replay_buffer=replay_buffer,
                online_network=online_network,
                target_network=target_network,
                optimizer=optimizer,
                random_generator=random.Random(7),
                warmup_size=4,
                batch_size=2,
                discount_factor=0.9,
                sync_interval=3,
            )
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    expected_steps = [4, 5, 6]
    expected_losses = [0.569000, 0.981785, 0.481624]
    expected_sync_flags = [False, False, True]
    trained_steps_ok = trained_steps == expected_steps
    losses_ok = len(losses) == 3 and all(
        abs(actual - expected) < 1e-5
        for actual, expected in zip(losses, expected_losses)
    )
    sync_flags_ok = sync_flags == expected_sync_flags
    update_count_ok = update_count == 3
    final_buffer_size_ok = len(replay_buffer) == 6
    online_changed = any(
        not torch.equal(old, current)
        for old, current in zip(
            online_before, online_network.parameters()
        )
    )
    final_target_matches = parameters_match(
        online_network, target_network
    )

    invalid_buffer = VectorReplayBuffer(capacity=6)
    invalid_online, invalid_target, invalid_optimizer = (
        make_training_objects()
    )
    invalid_warmup_rejected_before_collection = False
    try:
        collect_then_train(
            transition_stream=make_transition_stream(),
            replay_buffer=invalid_buffer,
            online_network=invalid_online,
            target_network=invalid_target,
            optimizer=invalid_optimizer,
            random_generator=random.Random(7),
            warmup_size=1,
            batch_size=2,
            discount_factor=0.9,
            sync_interval=3,
        )
    except ValueError:
        invalid_warmup_rejected_before_collection = (
            len(invalid_buffer) == 0
        )

    loss_by_step = dict(zip(trained_steps, losses))
    sync_by_step = dict(zip(trained_steps, sync_flags))
    update_number = 0
    for env_step in range(1, 7):
        trained = env_step in loss_by_step
        if trained:
            update_number += 1
            print(
                f"env_step={env_step} "
                f"buffer_size={env_step} "
                f"trained=True update_number={update_number} "
                f"loss={loss_by_step[env_step]:.6f} "
                f"synced={sync_by_step[env_step]}"
            )
        else:
            print(
                f"env_step={env_step} "
                f"buffer_size={env_step} "
                "trained=False update_number=0"
            )

    checks = (
        (f"trained_at_env_steps={expected_steps}", trained_steps_ok),
        ("three_losses_match_reference=True", losses_ok),
        (f"sync_flags={expected_sync_flags}", sync_flags_ok),
        ("update_count=3", update_count_ok),
        ("final_buffer_size=6", final_buffer_size_ok),
        ("online_parameters_changed=True", online_changed),
        ("final_target_matches_online=True", final_target_matches),
        (
            "invalid_warmup_rejected_before_collection=True",
            invalid_warmup_rejected_before_collection,
        ),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：先收集 4 条经验，再开始回放更新")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "collect_then_train() 中的 TODO"
        )


if __name__ == "__main__":
    main()
