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

本题不要求猜 `torch.tensor` 的接口。完整调用形式是：

    tensor = torch.tensor(data, dtype=数据类型)

- `data` 是必须提供的 Python 数字、列表或嵌套列表；
- `dtype=` 是具名参数，告诉 PyTorch 用什么类型保存这些数字；
- 不要写空的 `torch.tensor()`；
- 赋值行末不要多写逗号，否则得到的是 Python 元组而不是张量。

`transitions_to_tensors()` 已经用普通 `for` 循环完整示范 observation 字段：

    observations_data = []
    for transition in transitions:
        observations_data.append(transition.observation)

这段代码的意思是：先建空列表；每次从 `transitions` 取一条经验，
临时叫做 `transition`；读取它的 `.observation` 属性；再用 `append`
放到新列表末尾。

下面的列表推导式只是上述普通循环的短写，Python 3.6 已经支持：

    observations_data = [
        transition.observation for transition in transitions
    ]

你不需要猜或强制使用短写。只修改剩余 TODO：

1. 照 observations 的两步模式，分别收集其余四个同名字段；
2. actions 使用 `torch.long`；
3. rewards 使用 `torch.float32`；
4. next_observations 使用 `torch.float32`；
5. terminated 使用 `torch.bool`；
6. 按函数返回类型中的顺序返回五个张量。

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

    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
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
    # 已完成示范：先用普通 for 循环收集 observation。
    observations_data = []
    for transition in transitions:
        observations_data.append(transition.observation)

    # 然后才把 Python 列表转为张量。
    observations = torch.tensor(
        observations_data, dtype=torch.float32
    )

    actions_data = [
        transition.action for transition in transitions
    ]

    actions = torch.tensor(
        actions_data, dtype=torch.long
    )

    rewards_data = [
        transition.reward for transition in transitions
    ]

    rewards = torch.tensor(
        rewards_data, dtype=torch.float32
    )

    next_observations_data = [
        transition.next_observation for transition in transitions
    ]

    next_observations = torch.tensor(
        next_observations_data, dtype=torch.float32
    )

    terminateds_data = [
        transition.terminated for transition in transitions
    ]

    terminateds = torch.tensor(
        terminateds_data, dtype=torch.bool
    )

    truncateds_data = [
        transition.truncated for transition in transitions
    ]

    truncateds = torch.tensor(
        truncateds_data, dtype=torch.bool
    )

    return (
        observations, actions, rewards, next_observations, terminateds
    )


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


def show_tensor_api_example() -> None:
    """先打印已经提供的 observation 转换，让接口和形状可见。"""
    transitions = make_transitions()
    observations_data = [
        transition.observation for transition in transitions
    ]
    observations = torch.tensor(
        observations_data, dtype=torch.float32
    )
    print("已提供的 torch.tensor 接口示范")
    print(f"observations_data={observations_data}")
    print(f"observations=\n{observations}")
    print(f"observations.shape={tuple(observations.shape)}")
    print(f"observations.dtype={observations.dtype}")


def main() -> None:
    show_tensor_api_example()
    print()
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
