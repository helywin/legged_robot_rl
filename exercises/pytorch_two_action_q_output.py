#!/usr/bin/env python3
"""第 46 课编程练习：用 nn.Linear 输出两个动作的 Q 值。

为什么有这道题
============

前面的 PyTorch 模型只有一个观察量和一个输出。CartPole 的策略最终需要根据一组
观察，同时估计“向左推”和“向右推”两个动作的 Q 值。本题先缩小为两个观察量、
两个动作输出，不连接环境，也不训练模型。

场景
====

观察是长度为 2 的张量：

    observation = [位置, 变化趋势]

动作顺序固定为：

    0 = LEFT
    1 = RIGHT

模型必须返回长度为 2 的张量：

    q_values = [LEFT 的 Q 值, RIGHT 的 Q 值]

已有接口
========

创建并调用模型：

    model = TwoActionQExercise()
    q_values = model(observation)

检查器会在模型创建后写入固定参数，因此你不用计算或填写权重数值。

你的任务
========

只修改 `TwoActionQExercise` 中的两个 TODO：

1. 在 `__init__()` 中创建一个输入数量为 2、输出数量为 2 的 `nn.Linear`，保存为
   `self.q_values`；
2. 在 `forward()` 中把 observation 交给该线性层并返回结果。

带类型的 `__call__()` 已经提供，不要修改它。不要添加 softmax：Q 值不是概率，
可以为负数。不要修改 `check_exercise()`、`TEST_CASES` 或 `main()`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_two_action_q_output

成功条件
========

程序会检查：

    linear_layer_registered=True PASS
    parameter_shapes=(2, 2)/(2,) PASS
    三组 q_values 和 best_action 全部 PASS
    parameters_unchanged=True PASS

最后显示：

    练习通过：两个观察量已映射为两个动作 Q 值

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

本题只完成前向预测，没有选择训练样本中的实际动作、计算该动作的损失、更新参数、
连接 CartPole 或训练 DQN。
"""

from __future__ import annotations

import torch
from torch import nn


class TwoActionQExercise(nn.Module):
    """将两个观察量映射为 [LEFT, RIGHT] 两个 Q 值。"""

    def __init__(self) -> None:
        super().__init__()
        self.q_values = nn.Linear(2, 2)


    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """保留 Module 调用流程，并让 VS Code 推断返回 Tensor。"""
        return super().__call__(observation)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.q_values(observation)


TEST_CASES = (
    ([0.2, -0.1], [0.1, -0.05], 0),
    ([-1.0, 0.0], [-0.9, 1.2], 1),
    ([0.0, 0.0], [0.1, 0.2], 1),
)


def check_exercise() -> bool | None:
    """检查线性层、参数形状、两动作输出和参数不变性。"""
    try:
        model = TwoActionQExercise()
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    layer = getattr(model, "q_values", None)
    linear_layer_registered = isinstance(layer, nn.Linear)
    print(
        f"linear_layer_registered={linear_layer_registered} "
        f"{'PASS' if linear_layer_registered else 'FAIL'}"
    )
    if not linear_layer_registered:
        return False

    parameter_shapes = (
        tuple(layer.weight.shape) == (2, 2)
        and layer.bias is not None
        and tuple(layer.bias.shape) == (2,)
    )
    print(
        f"parameter_shapes={tuple(layer.weight.shape)}/"
        f"{tuple(layer.bias.shape) if layer.bias is not None else None} "
        f"{'PASS' if parameter_shapes else 'FAIL'}"
    )
    if not parameter_shapes:
        return False

    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32
            )
        )
        layer.bias.copy_(torch.tensor([0.1, 0.2], dtype=torch.float32))

    before = [parameter.detach().clone() for parameter in model.parameters()]
    all_passed = True
    for index, (values, expected_values, expected_best) in enumerate(
        TEST_CASES, start=1
    ):
        observation = torch.tensor(values, dtype=torch.float32)
        try:
            q_values = model(observation)
        except NotImplementedError as error:
            print(f"练习尚未完成：{error}")
            return None

        expected = torch.tensor(expected_values, dtype=torch.float32)
        correct_shape = isinstance(q_values, torch.Tensor) and q_values.shape == (2,)
        correct_values = correct_shape and torch.allclose(q_values, expected)
        best_action = q_values.argmax().item() if correct_shape else None
        passed = correct_values and best_action == expected_best
        shown_values = (
            [round(value, 3) for value in q_values.tolist()]
            if correct_shape
            else repr(q_values)
        )
        print(
            f"case{index}: q_values={shown_values} "
            f"best_action={best_action} {'PASS' if passed else 'FAIL'}"
        )
        all_passed = all_passed and passed

    unchanged = all(
        torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )
    print(
        f"parameters_unchanged={unchanged} "
        f"{'PASS' if unchanged else 'FAIL'}"
    )
    return all_passed and unchanged


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：两个观察量已映射为两个动作 Q 值")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "TwoActionQExercise 中的两个 TODO"
        )


if __name__ == "__main__":
    main()
