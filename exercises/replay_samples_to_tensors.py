#!/usr/bin/env python3
"""第 54 课编程练习：把回放经验对象列表组装成五个批量张量。

为什么有这道题
============

回放缓冲区随机抽出的是 Python `Transition` 对象列表，神经网络需要规则的张量。
训练前必须保持抽样顺序，把旧观察、动作、奖励、下一观察和真正终止标记分别收集，
并转换为符合其计算职责的 shape 与 dtype。

固定样本
========

    第0条: observation=[0.2,-0.1], action=1, reward=0.2,
           next_observation=[0.4,0.2], terminated=False
    第1条: observation=[-0.3,0.4], action=0, reward=1.0,
           next_observation=[-0.2,0.3], terminated=True

期望批量：

    observations.shape = (2, 2), dtype=float32
    actions.shape = (2,), dtype=long
    rewards.shape = (2,), dtype=float32
    next_observations.shape = (2, 2), dtype=float32
    terminated.shape = (2,), dtype=bool

你的任务
========

只修改 `transitions_to_tensors()` 中的 TODO：

1. 使用列表推导式，按当前抽样顺序分别收集五个同名字段；
2. 每个字段分别调用 `torch.tensor(...)`；
3. 旧观察和下一观察使用 `torch.float32`；
4. 动作使用 `torch.long`；
5. 奖励使用 `torch.float32`；
6. 真正终止标记使用 `torch.bool`；
7. 按函数返回类型中的顺序返回五个张量。

不要修改、排序或删除输入经验，不要把所有字段混进同一个张量，不要把 truncated
并入 terminated，不要调用网络、loss、backward 或 optimizer，也不要修改检查器。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.replay_samples_to_tensors

成功条件
========

程序应显示：

    observations_shape=(2, 2) dtype=float32 PASS
    actions_shape=(2,) dtype=long PASS
    rewards_shape=(2,) dtype=float32 PASS
    next_observations_shape=(2, 2) dtype=float32 PASS
    terminated_shape=(2,) dtype=bool PASS
    field_values_preserve_row_alignment=True PASS
    single_item_keeps_batch_dimension=True PASS
    input_transitions_unchanged=True PASS

最后显示：

    练习通过：回放样本已按字段组装成批量张量

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏仓库全量测试。

当前没有验证
============

本题没有执行随机抽样、网络前向、target、loss、backward、optimizer、CartPole 完整
训练、检查点或独立评估。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VectorTransition:
    """一次具有向量观察的环境交互记录。"""

    observation: tuple[float, float]
    action: int
    reward: float
    next_observation: tuple[float, float]
    terminated: bool
    truncated: bool


def transitions_to_tensors(
    transitions: list[VectorTransition],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """保持行对齐，把回放样本的五个字段分别转换为张量。"""
    # TODO: 只修改这里，按字段收集并使用正确 dtype 创建五个张量。
    raise NotImplementedError("请完成 transitions_to_tensors() 中的 TODO")


def make_transitions() -> list[VectorTransition]:
    """返回与前几课固定数字一致的两条经验。"""
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
    ]


def check_exercise() -> bool | None:
    """检查五个字段的 shape、dtype、行对齐和输入不变性。"""
    transitions = make_transitions()
    transitions_before = list(transitions)

    try:
        observations, actions, rewards, next_observations, terminated = (
            transitions_to_tensors(transitions)
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    observations_ok = (
        isinstance(observations, torch.Tensor)
        and observations.shape == (2, 2)
        and observations.dtype == torch.float32
    )
    actions_ok = (
        isinstance(actions, torch.Tensor)
        and actions.shape == (2,)
        and actions.dtype == torch.long
    )
    rewards_ok = (
        isinstance(rewards, torch.Tensor)
        and rewards.shape == (2,)
        and rewards.dtype == torch.float32
    )
    next_observations_ok = (
        isinstance(next_observations, torch.Tensor)
        and next_observations.shape == (2, 2)
        and next_observations.dtype == torch.float32
    )
    terminated_ok = (
        isinstance(terminated, torch.Tensor)
        and terminated.shape == (2,)
        and terminated.dtype == torch.bool
    )
    values_ok = all(
        (
            observations_ok,
            actions_ok,
            rewards_ok,
            next_observations_ok,
            terminated_ok,
        )
    ) and (
        torch.allclose(
            observations,
            torch.tensor([[0.2, -0.1], [-0.3, 0.4]]),
        )
        and torch.equal(actions, torch.tensor([1, 0]))
        and torch.allclose(rewards, torch.tensor([0.2, 1.0]))
        and torch.allclose(
            next_observations,
            torch.tensor([[0.4, 0.2], [-0.2, 0.3]]),
        )
        and torch.equal(terminated, torch.tensor([False, True]))
    )

    single = [transitions[0]]
    single_tensors = transitions_to_tensors(single)
    single_shapes_ok = tuple(
        tuple(tensor.shape) for tensor in single_tensors
    ) == ((1, 2), (1,), (1,), (1, 2), (1,))
    inputs_unchanged = (
        transitions == transitions_before
        and len(transitions) == 2
        and transitions[0].action == 1
        and transitions[1].action == 0
    )

    checks = (
        ("observations_shape=(2, 2) dtype=float32", observations_ok),
        ("actions_shape=(2,) dtype=long", actions_ok),
        ("rewards_shape=(2,) dtype=float32", rewards_ok),
        (
            "next_observations_shape=(2, 2) dtype=float32",
            next_observations_ok,
        ),
        ("terminated_shape=(2,) dtype=bool", terminated_ok),
        ("field_values_preserve_row_alignment=True", values_ok),
        ("single_item_keeps_batch_dimension=True", single_shapes_ok),
        ("input_transitions_unchanged=True", inputs_unchanged),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：回放样本已按字段组装成批量张量")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "transitions_to_tensors() 中的 TODO"
        )


if __name__ == "__main__":
    main()
