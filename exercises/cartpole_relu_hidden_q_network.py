#!/usr/bin/env python3
"""第 61 课编程练习：观察 ReLU 前后数据怎样流过 CartPole Q 网络。

为什么有这道题
================

两个线性层直接相连，整体仍可以合并成一个线性变换。ReLU 放在中间后，会把
负的隐藏值变成 0，使不同输入启用不同的隐藏单元，整体不再是一条全局直线。

本题不只要求创建层，还要求返回三个阶段的中间张量：

    observations
    → hidden_pre_activation
    → ReLU
    → hidden_activation
    → output_layer
    → q_values

固定网络
========

结构固定为：

    4 项观察 → 3 个隐藏单元 → 2 个动作 Q 值

对应层：

    hidden_layer = nn.Linear(in_features=4, out_features=3)
    activation = nn.ReLU()
    output_layer = nn.Linear(in_features=3, out_features=2)

参数形状与数量：

    hidden weight=(3, 4), bias=(3,)  → 12 + 3 = 15
    output weight=(2, 3), bias=(2,)  →  6 + 2 = 8
    总参数数目 = 23

检查器会写入固定参数。三条输入的预期中间结果是：

    hidden_pre_activation = [
        [ 0.2, -0.5, 0.2],
        [ 0.0,  0.0, 0.1],
        [-0.2,  0.5, 0.0],
    ]

ReLU 逐项执行 `max(0, value)`：

    hidden_activation = [
        [0.2, 0.0, 0.2],
        [0.0, 0.0, 0.1],
        [0.0, 0.5, 0.0],
    ]

最终：

    q_values = [
        [0.1, 0.3 ],
        [0.0, 0.3 ],
        [1.1, 0.35],
    ]

注意第一行隐藏值 `-0.5` 和第三行 `-0.2` 在 ReLU 后必须变成 0；正数保持
原值。最终输出层不加 ReLU，检查器会用另一条输入确认 Q 值仍然可以为负。

已知 PyTorch 接口
================

三个模块的完整构造接口已在上面给出。属性类型声明也已提供，避免 Pylance 把
未完成属性推断成 `Tensor | Module`。

调用顺序：

    hidden_pre_activation = self.hidden_layer(observations)
    hidden_activation = self.activation(hidden_pre_activation)
    q_values = self.output_layer(hidden_activation)

`forward_stages()` 必须返回：

    return hidden_pre_activation, hidden_activation, q_values

普通 `forward()` 仍只返回调用者真正需要的 `q_values`。可以调用
`forward_stages()`，用三个名字接收返回值，再返回第三项。

你的任务
========

只修改 `CartPoleReluQNetwork` 中的 TODO：

1. 创建并保存 `hidden_layer`、`activation`、`output_layer`；
2. 在 `forward_stages()` 中严格按照“隐藏线性层 → ReLU → 输出线性层”计算；
3. 返回 ReLU 前、ReLU 后和最终 Q 值三个张量；
4. 在 `forward()` 中复用 `forward_stages()`，但只返回最终 Q 值。

不要手写固定中间值，不要交换 ReLU 和线性层顺序，不要遗漏 ReLU，也不要在
最终 `q_values` 后再调用 ReLU。不要修改固定参数或检查器。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.cartpole_relu_hidden_q_network

成功条件
========

检查器会验证三个模块、四组参数形状、23 个参数、三个阶段的固定数值、负隐藏值
被截断、正隐藏值保留、批量动作、最终 Q 值允许为负、`forward()` 与分阶段结果
一致，以及前向计算不修改参数。最后应显示：

    练习通过：ReLU 隐藏层已形成可见的分段计算路径

TODO 未完成时只显示友好提示，不会输出异常堆栈或破坏全量测试。

当前没有验证
============

本题使用固定参数观察前向路径，没有 loss、backward、optimizer、探索、CartPole
完整训练、检查点、独立评估、Isaac Lab 或真机验证。隐藏宽度 3 只为方便手算，
不是声称真实任务的最佳宽度。
"""

from __future__ import annotations

import torch
from torch import nn


class CartPoleReluQNetwork(nn.Module):
    """四项观察经过三个 ReLU 隐藏单元，输出两个动作 Q 值。"""

    hidden_layer: nn.Linear
    activation: nn.ReLU
    output_layer: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        # 1：创建 4→3 的 hidden_layer、nn.ReLU activation、
        # 3→2 的 output_layer。
        self.hidden_layer = nn.Linear(4, 3)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(3, 2)

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        """保留 Module 调用流程，并让 VSCode 推断返回 Tensor。"""
        return super().__call__(observations)

    def forward_stages(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 ReLU 前、ReLU 后和最终 Q 值。"""
        # 2：依次调用 hidden_layer、activation、output_layer。
        v1 = self.hidden_layer(observations)
        v2 = self.activation(v1)
        v3 = self.output_layer(v2)
        # 3：按类型注解顺序返回三个阶段的张量。
        return (v1, v2, v3)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """对正常调用者只返回最终两个动作 Q 值。"""
        # 4：复用 forward_stages()，只返回第三项 q_values。
        v1 = self.hidden_layer(observations)
        v2 = self.activation(v1)
        v3 = self.output_layer(v2)
        return v3


def set_reference_parameters(model: CartPoleReluQNetwork) -> None:
    """写入课程中可手算的固定参数。"""
    with torch.no_grad():
        model.hidden_layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, -1.0, 0.0],
                    [-1.0, 0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        model.hidden_layer.bias.copy_(
            torch.tensor([0.0, 0.0, 0.1], dtype=torch.float32)
        )
        model.output_layer.weight.copy_(
            torch.tensor(
                [[1.0, 2.0, -1.0], [-1.0, 0.5, 2.0]],
                dtype=torch.float32,
            )
        )
        model.output_layer.bias.copy_(
            torch.tensor([0.1, 0.1], dtype=torch.float32)
        )


def check_exercise() -> bool | None:
    """检查结构、三个阶段的数据、输出边界和参数不变性。"""
    try:
        model = CartPoleReluQNetwork()
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    modules_ok = (
        isinstance(getattr(model, "hidden_layer", None), nn.Linear)
        and isinstance(getattr(model, "activation", None), nn.ReLU)
        and isinstance(getattr(model, "output_layer", None), nn.Linear)
    )
    if not modules_ok:
        print("three_modules_registered=False FAIL")
        return False

    shapes_ok = (
        model.hidden_layer.weight.shape == (3, 4)
        and model.hidden_layer.bias is not None
        and model.hidden_layer.bias.shape == (3,)
        and model.output_layer.weight.shape == (2, 3)
        and model.output_layer.bias is not None
        and model.output_layer.bias.shape == (2,)
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    set_reference_parameters(model)
    before = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    observations = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-0.2, 0.1, -0.3, 0.0],
        ],
        dtype=torch.float32,
    )
    try:
        hidden_pre, hidden_active, q_values = model.forward_stages(
            observations
        )
        normal_forward_q_values = model(observations)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    expected_hidden_pre = torch.tensor(
        [[0.2, -0.5, 0.2], [0.0, 0.0, 0.1], [-0.2, 0.5, 0.0]],
        dtype=torch.float32,
    )
    expected_hidden_active = torch.tensor(
        [[0.2, 0.0, 0.2], [0.0, 0.0, 0.1], [0.0, 0.5, 0.0]],
        dtype=torch.float32,
    )
    expected_q_values = torch.tensor(
        [[0.1, 0.3], [0.0, 0.3], [1.1, 0.35]],
        dtype=torch.float32,
    )
    hidden_pre_ok = (
        hidden_pre.shape == (3, 3)
        and torch.allclose(hidden_pre, expected_hidden_pre)
    )
    hidden_active_ok = (
        hidden_active.shape == (3, 3)
        and torch.allclose(hidden_active, expected_hidden_active)
    )
    relu_rule_ok = (
        torch.all(hidden_active >= 0).item()
        and hidden_active[0, 1].item() == 0.0
        and hidden_active[2, 0].item() == 0.0
        and hidden_active[0, 0].item() == hidden_pre[0, 0].item()
    )
    q_values_ok = (
        q_values.shape == (3, 2)
        and torch.allclose(q_values, expected_q_values)
    )
    best_actions = q_values.argmax(dim=1)
    best_actions_ok = torch.equal(
        best_actions, torch.tensor([1, 1, 0])
    )
    forward_matches_stages = torch.allclose(
        normal_forward_q_values, q_values
    )

    negative_output = model(
        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    )
    output_can_be_negative = negative_output[0].item() < 0.0
    parameters_unchanged = all(
        torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )

    checks = (
        ("three_modules_registered=True", modules_ok),
        ("parameter_shapes=(3,4)/(3,)/(2,3)/(2,)", shapes_ok),
        ("parameter_count=23", parameter_count == 23),
        ("hidden_pre_activation_matches=True", hidden_pre_ok),
        ("hidden_activation_matches=True", hidden_active_ok),
        ("relu_clamps_negative_and_keeps_positive=True", relu_rule_ok),
        ("q_values_match=True", q_values_ok),
        ("best_actions=[1, 1, 0]", best_actions_ok),
        ("forward_matches_forward_stages=True", forward_matches_stages),
        ("final_q_values_can_be_negative=True", output_can_be_negative),
        ("parameters_unchanged=True", parameters_unchanged),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：ReLU 隐藏层已形成可见的分段计算路径")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "CartPoleReluQNetwork 中的 TODO"
        )


if __name__ == "__main__":
    main()
