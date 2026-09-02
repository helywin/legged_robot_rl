#!/usr/bin/env python3
"""第 55 课编程练习：把回放抽样、张量组装和批量更新连起来。

为什么有这道题
================

第 53 课的更新函数直接接收五个张量，第 54 课已经把经验对象列表变成
这五个张量。真实的一次回放更新需要把三段数据流接起来：

    ReplayBuffer.sample(...)
    -> transitions_to_tensors(...)
    -> full_batch_dqn_update(...)

本题不要求重写任何 DQN 公式。

固定数据
========

缓冲区保存三条经验：

    位置0: observation=( 0.2,-0.1), action=1, reward= 0.2
    位置1: observation=(-0.3, 0.4), action=0, reward= 1.0
    位置2: observation=( 0.0, 0.0), action=0, reward=-0.2

`random.Random(7)` 抽取两条时，顺序固定为位置 1、0。所以进入更新函数的
核心数字应为：

    observations      = [[-0.3, 0.4], [0.2, -0.1]]
    actions           = [0, 1]
    rewards           = [1.0, 0.2]
    next_observations = [[-0.2, 0.3], [0.4, 0.2]]
    terminated        = [True, False]

三个已知接口
============

1. 抽样接口：

       sampled_transitions = replay_buffer.sample(
           sample_size=batch_size,
           rng=random_generator,
       )

   输入是抽样数量和随机数对象，返回 `list[VectorTransition]`。

2. 张量组装接口：

       tensors = transitions_to_tensors(sampled_transitions)

   返回顺序固定为：

       (
           observations,
           actions,
           rewards,
           next_observations,
           terminated,
       )

   可以用五个变量直接解包，也可以先保存到 `tensors` 再用索引取值。

3. 批量更新接口：

       result = full_batch_dqn_update(
           online_network,
           target_network,
           optimizer,
           observations,
           actions,
           rewards,
           next_observations,
           terminated,
           discount_factor,
       )

   返回五个用来检查本次更新的张量。`train_from_replay()` 直接返回这个
   `result` 即可，不要自己再计算一遍 loss。

你的任务
========

只修改 `train_from_replay()` 中的 TODO：

1. 调用已给出的抽样接口；
2. 把抽样列表交给张量组装函数，按固定顺序取出五个张量；
3. 把张量和原有训练对象交给批量更新函数；
4. 返回批量更新函数的结果。

不要修改抽样算法、重新排序样本、重写张量组装、复制 DQN 公式、同步目标
网络或修改检查器。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.replay_sample_batch_update

成功条件
========

程序最后应显示：

    sampled_row_order_is_1_then_0=True PASS
    sampled_fields_keep_row_alignment=True PASS
    mean_loss=0.6418 PASS
    replay_buffer_unchanged=True PASS
    online_parameters_changed=True PASS
    target_parameters_unchanged=True PASS

并显示：

    练习通过：回放抽样已接入一次批量 DQN 更新

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏仓库全量测试。

当前没有验证
============

本题没有环境 `step()`、完整 episode、目标网络定期同步、CartPole 冒烟训练、
检查点、独立评估、Isaac Lab 或真机验证。
"""

from __future__ import annotations

import random

import torch

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)
from exercises.pytorch_full_batch_dqn_update import (
    full_batch_dqn_update,
)
from exercises.replay_samples_to_tensors import (
    VectorTransition,
    transitions_to_tensors,
)


class VectorReplayBuffer:
    """保存向量观察经验；本课学习者不需要修改这个类。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity 必须至少为 1")
        self._capacity = capacity
        self._transitions: list[VectorTransition] = []

    def __len__(self) -> int:
        return len(self._transitions)

    def add(self, transition: VectorTransition) -> None:
        """保存一条经验；容量满时先淘汰最旧的一条。"""
        if len(self._transitions) == self._capacity:
            self._transitions.pop(0)
        self._transitions.append(transition)

    def sample(
        self, sample_size: int, rng: random.Random
    ) -> list[VectorTransition]:
        """随机抽取互不重复的经验，不从缓冲区删除它们。"""
        if sample_size < 1:
            raise ValueError("sample_size 必须至少为 1")
        if sample_size > len(self._transitions):
            raise ValueError("sample_size 不能大于当前经验数量")
        return rng.sample(self._transitions, sample_size)

    def snapshot(self) -> tuple[VectorTransition, ...]:
        """返回当前内容快照，用于观察抽样前后是否不变。"""
        return tuple(self._transitions)


def train_from_replay(
    replay_buffer: VectorReplayBuffer,
    online_network: TwoActionQModule,
    target_network: TwoActionQModule,
    optimizer: torch.optim.Optimizer,
    random_generator: random.Random,
    batch_size: int,
    discount_factor: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """抽样、组装张量，再完成一次在线网络更新。"""
    # TODO 1：调用 replay_buffer.sample(...)。
    # 完整接口已写在文件顶部，返回经验对象列表。

    # TODO 2：调用 transitions_to_tensors(...)。
    # 按 observations、actions、rewards、next_observations、
    # terminated 的顺序取出五个张量。

    # TODO 3：调用 full_batch_dqn_update(...)。
    # 传入原有三个训练对象、五个张量和 discount_factor，
    # 直接返回它的返回值。
    raise NotImplementedError(
        "请按文件顶部的完整接口依次连接三个函数"
    )


def make_replay_buffer() -> VectorReplayBuffer:
    """创建包含三条固定经验的缓冲区。"""
    replay_buffer = VectorReplayBuffer(capacity=3)
    replay_buffer.add(
        VectorTransition(
            observation=(0.2, -0.1),
            action=1,
            reward=0.2,
            next_observation=(0.4, 0.2),
            terminated=False,
            truncated=False,
        )
    )
    replay_buffer.add(
        VectorTransition(
            observation=(-0.3, 0.4),
            action=0,
            reward=1.0,
            next_observation=(-0.2, 0.3),
            terminated=True,
            truncated=False,
        )
    )
    replay_buffer.add(
        VectorTransition(
            observation=(0.0, 0.0),
            action=0,
            reward=-0.2,
            next_observation=(0.1, 0.0),
            terminated=False,
            truncated=False,
        )
    )
    return replay_buffer


def make_training_objects() -> tuple[
    TwoActionQModule,
    TwoActionQModule,
    torch.optim.Optimizer,
]:
    """创建并固定两个网络参数，使本课结果可以手算。"""
    online_network = TwoActionQModule()
    target_network = TwoActionQModule()
    set_demo_parameters(online_network)
    set_demo_parameters(target_network)
    optimizer = torch.optim.SGD(online_network.parameters(), lr=0.1)
    return online_network, target_network, optimizer


def show_pre_update_data_flow() -> None:
    """在 TODO 前就让抽样顺序和张量数据可见。"""
    replay_buffer = make_replay_buffer()
    sampled = replay_buffer.sample(
        sample_size=2, rng=random.Random(7)
    )
    observations, actions, rewards, next_observations, terminated = (
        transitions_to_tensors(sampled)
    )

    stored_observations = []
    for transition in replay_buffer.snapshot():
        stored_observations.append(transition.observation)

    sampled_observations = []
    for transition in sampled:
        sampled_observations.append(transition.observation)

    print("已提供的更新前数据流")
    print(f"buffer_observations={stored_observations}")
    print(f"sampled_observations={sampled_observations}")
    print(f"observations=\n{observations}")
    print(f"actions={actions}")
    print(f"rewards={rewards}")
    print(f"next_observations=\n{next_observations}")
    print(f"terminated={terminated}")


def check_exercise() -> bool | None:
    """检查抽样顺序、行对齐、更新结果和职责边界。"""
    replay_buffer = make_replay_buffer()
    online_network, target_network, optimizer = make_training_objects()
    buffer_before = replay_buffer.snapshot()
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
            train_from_replay(
                replay_buffer=replay_buffer,
                online_network=online_network,
                target_network=target_network,
                optimizer=optimizer,
                random_generator=random.Random(7),
                batch_size=2,
                discount_factor=0.9,
            )
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    expected_online_q = torch.tensor(
        [[0.6, 0.7], [0.1, -0.05]], dtype=torch.float32
    )
    expected_selected = torch.tensor(
        [0.6, -0.05], dtype=torch.float32
    )
    expected_targets = torch.tensor([1.0, 1.01], dtype=torch.float32)
    expected_losses = torch.tensor(
        [0.16, 1.1236], dtype=torch.float32
    )

    sampled_order_ok = (
        isinstance(online_q, torch.Tensor)
        and online_q.shape == (2, 2)
        and torch.allclose(online_q, expected_online_q)
    )
    row_alignment_ok = (
        isinstance(selected_q, torch.Tensor)
        and isinstance(targets, torch.Tensor)
        and isinstance(per_item_losses, torch.Tensor)
        and torch.allclose(selected_q, expected_selected)
        and torch.allclose(targets, expected_targets)
        and torch.allclose(per_item_losses, expected_losses)
    )
    loss_ok = (
        isinstance(loss, torch.Tensor)
        and loss.ndim == 0
        and abs(loss.item() - 0.6418) < 1e-6
    )
    buffer_unchanged = replay_buffer.snapshot() == buffer_before
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
        ("sampled_row_order_is_1_then_0=True", sampled_order_ok),
        ("sampled_fields_keep_row_alignment=True", row_alignment_ok),
        ("mean_loss=0.6418", loss_ok),
        ("replay_buffer_unchanged=True", buffer_unchanged),
        ("online_parameters_changed=True", online_changed),
        ("target_parameters_unchanged=True", target_unchanged),
        ("target_gradients_none=True", target_gradients_none),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    show_pre_update_data_flow()
    print()
    result = check_exercise()
    if result:
        print()
        print(
            "练习通过：回放抽样已接入一次批量 DQN 更新"
        )
    elif result is False:
        print()
        print("还有检查未通过，请只修改 train_from_replay() 中的 TODO")


if __name__ == "__main__":
    main()
