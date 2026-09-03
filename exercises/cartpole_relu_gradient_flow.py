#!/usr/bin/env python3
"""第 62 课编程练习：追踪梯度怎样穿过输出层和 ReLU。

为什么有这道题
================

上一课只做了前向计算：观察经过隐藏层、ReLU 和输出层得到两个 Q 值。本题加入
所选动作、目标值、均方误差和 ``backward()``，亲手检查梯度沿哪条路径返回。

固定前向数据
============

沿用第 61 课的固定网络，对一条观察：

    observation = [0.2, -0.1, 0.3, 0.0]

前向结果固定为：

    hidden_pre    = [ 0.2, -0.5, 0.2]
    hidden_active = [ 0.2,  0.0, 0.2]
    q_values      = [ 0.1,  0.3]

第一次检查使用实际动作 1 和目标 Q 值 0.7：

    selected_q = 0.3
    loss = (0.3 - 0.7) ** 2 = 0.16
    d(loss) / d(selected_q) = 2 * (0.3 - 0.7) = -0.8

输出层动作 1 的权重是 ``[-1.0, 0.5, 2.0]``，因此 ReLU 输出处的梯度是：

    hidden_active_grad = -0.8 * [-1.0, 0.5, 2.0]
                       = [0.8, -0.4, -1.6]

第二个隐藏单元的 ReLU 输入是 -0.5。它的局部导数为 0，所以梯度继续穿过
ReLU 后变为：

    hidden_pre_grad = [0.8, 0.0, -1.6]

这里会同时出现三类零梯度：

1. 动作 0 没被 loss 选中，所以输出层第 0 行梯度为 0；
2. 隐藏单元 1 的 ReLU 输入为负，所以隐藏层第 1 行梯度为 0；
3. observation[3] 等于 0，所以隐藏层权重梯度的第 3 列为 0。

已知 PyTorch 接口
================

这不是接口猜谜。完成 TODO 会用到的调用全部列在这里：

``model.zero_grad(set_to_none=True)``
    在本次反向传播前清除模型参数的旧梯度。``set_to_none=True`` 会把旧
    ``.grad`` 设回 ``None``，而不是保留一块全零张量。

``model.forward_stages(observation)``
    返回 ``hidden_pre, hidden_active, q_values``，三个张量仍属于同一张计算图。

``tensor.retain_grad()``
    必须在 ``backward()`` 之前调用。模型参数是叶子张量，PyTorch 默认保存它们
    的 ``.grad``；隐藏中间张量不是叶子，反传时会用到梯度，但默认不把梯度
    留在 ``.grad`` 中。``retain_grad()`` 只为观察保留它，不改变梯度算法。

``q_values[action_index]``
    按经验里实际执行的动作编号选择一个标量 Q 值，不使用 ``argmax()``。

``torch.tensor(target_q, dtype=selected_q.dtype)``
    创建与预测 dtype 相同的标量目标张量。

``torch.nn.functional.mse_loss(selected_q, target)``
    计算标量均方误差。本题只有一个数，所以就是
    ``(selected_q - target) ** 2``。

``loss.backward()``
    沿本轮前向形成的计算图反传，把结果写入参数和已经调用
    ``retain_grad()`` 的中间张量的 ``.grad``。

``required_gradient(tensor, label)``
    本文件已经提供的辅助函数。它先处理 Pylance 看到的 ``Tensor | None``，
    若梯度不存在就给出明确错误；存在时返回一份脱离计算图的副本。

你的任务
========

只修改 ``compute_gradient_trace()`` 中的 TODO：

1. 先验证 ``action_index`` 只能是 0 或 1；
2. 清除旧梯度；
3. 用一次 ``forward_stages()`` 得到三个前向阶段，不能手写固定结果；
4. 在反传前要求 PyTorch 保留两个隐藏中间张量的梯度；
5. 按实际动作取 ``selected_q``，创建 target，计算 MSE loss；
6. 调用一次 ``backward()``；
7. 用 ``required_gradient()`` 收集两个中间张量和两层参数的梯度；
8. 返回完整的 ``GradientTrace``。

检查器会连续测试动作 1 和动作 0。若把动作编号写死、忘记清梯度、对最终 Q 值
使用 ReLU，或把 ``retain_grad()`` 放在反传之后，都不能通过。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.cartpole_relu_gradient_flow

成功条件
========

检查器会验证前向结果、loss、两个动作的梯度切换、ReLU 截断、零输入特征、旧梯度
清除和参数不变性。最后应显示：

    练习通过：所选动作、ReLU 和输入值怎样控制梯度路径已经可见

TODO 未完成时只显示友好提示，不输出异常堆栈，也不会破坏仓库全量测试。

当前没有验证
============

本题只执行前向和反向传播，没有调用 optimizer，所以参数不会改变。它没有构造
DQN 的奖励与未来价值 target，没有完整 CartPole 训练、检查点、独立评估、
Isaac Lab 或真机验证。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from exercises.cartpole_relu_hidden_q_network import (
    CartPoleReluQNetwork,
    set_reference_parameters,
)


@dataclass(frozen=True)
class GradientTrace:
    """一次前向与反向传播中需要对照的数值。"""

    hidden_pre: torch.Tensor
    hidden_active: torch.Tensor
    q_values: torch.Tensor
    selected_q: torch.Tensor
    loss: torch.Tensor
    hidden_pre_grad: torch.Tensor
    hidden_active_grad: torch.Tensor
    hidden_weight_grad: torch.Tensor
    hidden_bias_grad: torch.Tensor
    output_weight_grad: torch.Tensor
    output_bias_grad: torch.Tensor


def required_gradient(tensor: torch.Tensor, label: str) -> torch.Tensor:
    """取得梯度副本，并把 Optional 类型变成 Pylance 明确的 Tensor。"""
    gradient = tensor.grad
    if gradient is None:
        raise RuntimeError(
            f"{label}.grad 仍是 None；请检查 retain_grad()、backward() 和计算路径"
        )
    return gradient.detach().clone()


def compute_gradient_trace(
    model: CartPoleReluQNetwork,
    observation: torch.Tensor,
    action_index: int,
    target_q: float,
) -> GradientTrace:
    """完成一次反向传播，并返回各阶段的数值与梯度。"""
    # TODO 1：先验证 action_index。只有动作 0 和动作 1 合法；其他值抛出
    # ValueError("action_index 必须是 0 或 1")。

    # TODO 2：清除模型参数上一次留下的梯度。

    # TODO 3：只调用一次 forward_stages(observation)，取得 hidden_pre、
    # hidden_active 和 q_values。

    # TODO 4：在 backward() 前，让两个隐藏中间张量保留 .grad。

    # TODO 5：按 action_index 选择标量 Q 值，创建相同 dtype 的 target，
    # 再用 F.mse_loss() 得到 loss。

    # TODO 6：调用一次 backward()。

    # TODO 7：构造并返回 GradientTrace。前向的五个字段使用 detach().clone()；
    # 六个梯度字段全部调用 required_gradient(tensor, "清楚的字段名")。
    raise NotImplementedError("请完成一次穿过 ReLU 的梯度追踪")


def _allclose(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return actual.shape == expected.shape and torch.allclose(
        actual,
        expected,
        atol=1e-6,
    )


def _run_trace(
    model: CartPoleReluQNetwork,
    observation: torch.Tensor,
    action_index: int,
    target_q: float,
) -> GradientTrace | None:
    try:
        return compute_gradient_trace(
            model,
            observation,
            action_index,
            target_q,
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None
    except RuntimeError as error:
        print(f"梯度追踪尚未接通：{error}")
        return None


def check_exercise() -> bool | None:
    """检查两种动作路径、三类零梯度、清零和参数不变性。"""
    model = CartPoleReluQNetwork()
    set_reference_parameters(model)
    observation = torch.tensor(
        [0.2, -0.1, 0.3, 0.0],
        dtype=torch.float32,
    )
    parameters_before = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 99.0)

    action_1 = _run_trace(model, observation, action_index=1, target_q=0.7)
    if action_1 is None:
        return None

    action_1_forward_ok = (
        _allclose(action_1.hidden_pre, torch.tensor([0.2, -0.5, 0.2]))
        and _allclose(action_1.hidden_active, torch.tensor([0.2, 0.0, 0.2]))
        and _allclose(action_1.q_values, torch.tensor([0.1, 0.3]))
        and _allclose(action_1.selected_q, torch.tensor(0.3))
        and _allclose(action_1.loss, torch.tensor(0.16))
    )
    action_1_intermediate_grad_ok = (
        _allclose(
            action_1.hidden_active_grad,
            torch.tensor([0.8, -0.4, -1.6]),
        )
        and _allclose(
            action_1.hidden_pre_grad,
            torch.tensor([0.8, 0.0, -1.6]),
        )
    )
    action_1_parameter_grad_ok = (
        _allclose(
            action_1.output_weight_grad,
            torch.tensor(
                [[0.0, 0.0, 0.0], [-0.16, 0.0, -0.16]]
            ),
        )
        and _allclose(
            action_1.output_bias_grad,
            torch.tensor([0.0, -0.8]),
        )
        and _allclose(
            action_1.hidden_weight_grad,
            torch.tensor(
                [
                    [0.16, -0.08, 0.24, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-0.32, 0.16, -0.48, 0.0],
                ]
            ),
        )
        and _allclose(
            action_1.hidden_bias_grad,
            torch.tensor([0.8, 0.0, -1.6]),
        )
    )
    old_gradient_was_cleared = not torch.any(
        action_1.output_weight_grad == 99.0
    ).item()

    action_0 = _run_trace(model, observation, action_index=0, target_q=0.5)
    if action_0 is None:
        return None

    action_switch_ok = (
        _allclose(action_0.selected_q, torch.tensor(0.1))
        and _allclose(action_0.loss, torch.tensor(0.16))
        and _allclose(
            action_0.output_weight_grad,
            torch.tensor(
                [[-0.16, 0.0, -0.16], [0.0, 0.0, 0.0]]
            ),
        )
        and _allclose(
            action_0.output_bias_grad,
            torch.tensor([-0.8, 0.0]),
        )
        and _allclose(
            action_0.hidden_pre_grad,
            torch.tensor([-0.8, 0.0, 0.8]),
        )
    )
    relu_gate_ok = (
        action_1.hidden_active_grad[1].item() != 0.0
        and action_1.hidden_pre_grad[1].item() == 0.0
        and torch.all(action_1.hidden_weight_grad[1] == 0.0).item()
        and torch.all(action_0.hidden_weight_grad[1] == 0.0).item()
    )
    input_zero_gate_ok = (
        torch.all(action_1.hidden_weight_grad[:, 3] == 0.0).item()
        and torch.all(action_0.hidden_weight_grad[:, 3] == 0.0).item()
    )
    parameters_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters())
    )

    invalid_action_rejected = False
    try:
        compute_gradient_trace(model, observation, action_index=2, target_q=0.0)
    except ValueError:
        invalid_action_rejected = True

    checks = [
        ("action_1_forward_and_loss", action_1_forward_ok),
        ("action_1_intermediate_gradients", action_1_intermediate_grad_ok),
        ("action_1_parameter_gradients", action_1_parameter_grad_ok),
        ("old_gradient_was_cleared", old_gradient_was_cleared),
        ("action_switch_changes_output_path", action_switch_ok),
        ("relu_blocks_inactive_hidden_path", relu_gate_ok),
        ("zero_input_blocks_weight_column", input_zero_gate_ok),
        ("invalid_action_rejected", invalid_action_rejected),
        ("backward_does_not_change_parameters", parameters_unchanged),
    ]
    for label, passed in checks:
        print(f"{label}={passed} {'PASS' if passed else 'FAIL'}")

    passed = all(result for _, result in checks)
    if passed:
        print(
            "\n练习通过：所选动作、ReLU 和输入值怎样控制梯度路径已经可见"
        )
    else:
        print("\n练习未通过：请对照上面的 FAIL 和文件顶部手算结果")
    return passed


if __name__ == "__main__":
    result = check_exercise()
    if result is False:
        raise SystemExit(1)
