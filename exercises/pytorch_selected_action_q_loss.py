#!/usr/bin/env python3
"""第 47 课编程练习：只用实际执行动作的 Q 值计算损失。

为什么有这道题
============

模型会同时输出 LEFT 和 RIGHT 两个 Q 值，但一条环境经验只告诉我们实际执行动作的
结果。因此训练这条经验时，应从全部 Q 值中取出实际动作对应的一个标量，再与目标
Q 值比较。

本题只学习“按动作编号取值并计算损失”，不更新参数，也不计算完整 DQN 目标。

场景
====

固定观察产生：

    q_values = [0.10, -0.05]

其中当前最大值对应动作 0，但这条经验实际执行的是动作 1：

    executed_action = 1
    target = 0.50

因此本次必须选择 `-0.05`，而不是最大值 `0.10`。平方损失为：

    (-0.05 - 0.50) ** 2 = 0.3025

已有接口
========

`selected_action_loss(model, observation, executed_action, target)` 必须返回：

    q_values, selected_q, loss

你的任务
========

只修改 `selected_action_loss()` 中的 TODO：

1. 调用模型得到全部动作的 q_values；
2. 使用 executed_action 作为张量索引，取得标量 selected_q；
3. 计算 selected_q 与 target 的平方损失；
4. 调用反向计算；
5. 返回 q_values、selected_q 和 loss。

不要使用 `argmax()` 代替 executed_action，不要把 selected_q 转成 Python 数值，
不要调用优化器或直接修改梯度，不要修改 `check_exercise()` 或 `main()`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_selected_action_q_loss

成功条件
========

程序应显示：

    q_values=[0.1, -0.05] PASS
    best_action=0 PASS
    executed_action=1 PASS
    selected_q=-0.050 PASS
    loss=0.3025 PASS
    unselected_row_grad_zero=True PASS
    selected_row_grad_nonzero=True PASS

最后显示：

    练习通过：损失只连接到实际执行动作的 Q 值

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

本题使用人工给定的 target，没有根据奖励和下一观察计算 DQN 目标，也没有调用
optimizer 更新参数、连接 CartPole 或训练完整 DQN。
"""

from __future__ import annotations

import torch

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


def selected_action_loss(
    model: TwoActionQModule,
    observation: torch.Tensor,
    executed_action: int,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回全部 Q 值、实际动作 Q 值和平方损失，并计算梯度。"""
    predict = model(observation)
    selected_q = predict[executed_action]
    loss = (target - selected_q) ** 2
    loss.backward()
    return (predict, selected_q, loss)


def check_exercise() -> bool | None:
    """检查选值、损失以及梯度只流向实际动作输出行。"""
    model = TwoActionQModule()
    set_demo_parameters(model)
    observation = torch.tensor([0.2, -0.1], dtype=torch.float32)
    executed_action = 1
    target = torch.tensor(0.5, dtype=torch.float32)

    try:
        q_values, selected_q, loss = selected_action_loss(
            model, observation, executed_action, target
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    q_values_ok = (
        isinstance(q_values, torch.Tensor)
        and q_values.shape == (2,)
        and torch.allclose(q_values, torch.tensor([0.1, -0.05]))
    )
    selected_q_ok = (
        isinstance(selected_q, torch.Tensor)
        and selected_q.ndim == 0
        and abs(selected_q.item() + 0.05) < 1e-6
    )
    loss_ok = (
        isinstance(loss, torch.Tensor)
        and loss.ndim == 0
        and abs(loss.item() - 0.3025) < 1e-6
    )
    best_action = q_values.argmax().item() if q_values_ok else None
    unselected_zero = (
        model.q_values.weight.grad is not None
        and model.q_values.bias.grad is not None
        and torch.equal(
            model.q_values.weight.grad[0], torch.tensor([0.0, 0.0])
        )
        and model.q_values.bias.grad[0].item() == 0.0
    )
    selected_nonzero = (
        model.q_values.weight.grad is not None
        and model.q_values.bias.grad is not None
        and torch.allclose(
            model.q_values.weight.grad[1], torch.tensor([-0.22, 0.11])
        )
        and abs(model.q_values.bias.grad[1].item() + 1.1) < 1e-6
    )

    checks = (
        ("q_values=[0.1, -0.05]", q_values_ok),
        ("best_action=0", best_action == 0),
        ("executed_action=1", executed_action == 1),
        ("selected_q=-0.050", selected_q_ok),
        ("loss=0.3025", loss_ok),
        ("unselected_row_grad_zero=True", unselected_zero),
        ("selected_row_grad_nonzero=True", selected_nonzero),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：损失只连接到实际执行动作的 Q 值")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "selected_action_loss() 中的 TODO"
        )


if __name__ == "__main__":
    main()
