#!/usr/bin/env python3
"""第 50 课编程练习：从一批 Q 值中逐行取得实际动作对应的值。

为什么有这道题
============

单条经验的 q_values 形状是 `(动作数量,)`，可以直接使用一个动作编号索引。经验回放
会抽出多条经验，此时网络输出形状变成 `(批量大小, 动作数量)`。每一行都有自己的
实际动作，不能再把动作编号误当成经验行号。

场景
====

两条观察一起进入固定的两动作网络：

    observations = [
        [ 0.2, -0.1],
        [-0.3,  0.4],
    ]

网络输出：

    q_values = [
        [0.10, -0.05],
        [0.60,  0.70],
    ]

两条经验实际执行的动作分别是：

    executed_actions = [1, 0]

所以第 0 行应取第 1 列，第 1 行应取第 0 列：

    selected_q_values = [-0.05, 0.60]

你的任务
========

只修改 `select_batch_action_q_values()` 中的 TODO：

1. 调用 model，让 observations 得到形状 `(2, 2)` 的 q_values；
2. 使用 `unsqueeze(1)` 把动作编号从 `(2,)` 变成逐行列号 `(2, 1)`；
3. 使用张量的 `gather(dim=1, index=...)`，从每一行取得指定动作列；
4. 使用 `squeeze(1)` 把结果从 `(2, 1)` 恢复为 `(2,)`；
5. 返回 q_values 和 selected_q_values。

不要使用 `argmax()`，不要使用 `q_values[executed_actions]`，不要逐条调用模型，
不要把结果转换成 Python list，也不要修改模型参数、`check_exercise()` 或 `main()`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_batch_selected_q

成功条件
========

程序应显示：

    q_values_shape=(2, 2) PASS
    q_values=[[0.1, -0.05], [0.6, 0.7]] PASS
    executed_actions=[1, 0] PASS
    selected_q_values=[-0.05, 0.6] PASS
    selected_shape=(2,) PASS
    single_item_batch_keeps_shape=(1,) PASS
    selected_keeps_gradient=True PASS
    parameters_unchanged=True PASS

最后显示：

    练习通过：一批经验已逐行取得实际动作 Q 值

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏仓库全量测试。

当前没有验证
============

本题没有计算 target、loss 或梯度，没有调用 optimizer，也没有从真实回放缓冲区抽样
或运行 CartPole。
"""

from __future__ import annotations

import torch

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


def select_batch_action_q_values(
    model: TwoActionQModule,
    observations: torch.Tensor,
    executed_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回整批全部动作 Q 值和每行实际动作对应的 Q 值。"""
    q_values = model(observations)
    indexes = executed_actions.unsqueeze(1)
    selected_q_values_vec = q_values.gather(dim=1, index=indexes)
    selected_q_values = selected_q_values_vec.squeeze(1)
    return (q_values, selected_q_values)


def check_exercise() -> bool | None:
    """检查批量形状、逐行动作索引、计算图和参数不变性。"""
    model = TwoActionQModule()
    set_demo_parameters(model)
    observations = torch.tensor(
        [[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float32
    )
    executed_actions = torch.tensor([1, 0], dtype=torch.long)
    parameters_before = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    try:
        q_values, selected_q_values = select_batch_action_q_values(
            model, observations, executed_actions
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    expected_q_values = torch.tensor(
        [[0.1, -0.05], [0.6, 0.7]], dtype=torch.float32
    )
    expected_selected = torch.tensor([-0.05, 0.6], dtype=torch.float32)
    q_shape_ok = (
        isinstance(q_values, torch.Tensor)
        and q_values.shape == (2, 2)
    )
    q_values_ok = q_shape_ok and torch.allclose(
        q_values, expected_q_values
    )
    selected_values_ok = (
        isinstance(selected_q_values, torch.Tensor)
        and selected_q_values.shape == (2,)
        and torch.allclose(selected_q_values, expected_selected)
    )
    selected_shape_ok = (
        isinstance(selected_q_values, torch.Tensor)
        and selected_q_values.shape == (2,)
    )
    single_observation = torch.tensor(
        [[0.2, -0.1]], dtype=torch.float32
    )
    single_action = torch.tensor([1], dtype=torch.long)
    _single_q_values, single_selected = select_batch_action_q_values(
        model, single_observation, single_action
    )
    single_batch_shape_ok = (
        isinstance(single_selected, torch.Tensor)
        and single_selected.shape == (1,)
        and torch.allclose(single_selected, torch.tensor([-0.05]))
    )
    keeps_gradient = (
        isinstance(selected_q_values, torch.Tensor)
        and selected_q_values.requires_grad
        and selected_q_values.grad_fn is not None
    )
    parameters_unchanged = all(
        torch.equal(old, current)
        for old, current in zip(parameters_before, model.parameters())
    )

    checks = (
        ("q_values_shape=(2, 2)", q_shape_ok),
        ("q_values=[[0.1, -0.05], [0.6, 0.7]]", q_values_ok),
        ("executed_actions=[1, 0]", executed_actions.tolist() == [1, 0]),
        ("selected_q_values=[-0.05, 0.6]", selected_values_ok),
        ("selected_shape=(2,)", selected_shape_ok),
        ("single_item_batch_keeps_shape=(1,)", single_batch_shape_ok),
        ("selected_keeps_gradient=True", keeps_gradient),
        ("parameters_unchanged=True", parameters_unchanged),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：一批经验已逐行取得实际动作 Q 值")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "select_batch_action_q_values() 中的 TODO"
        )


if __name__ == "__main__":
    main()
