#!/usr/bin/env python3
"""第 42 课编程练习：从预测误差得到参数梯度。

为什么有这道题
============

`nn.Module` 已经能用参数完成预测，但它还不知道预测应该怎样改变。本题给出一个
目标值，用平方损失表示预测与目标的差距，再调用 `backward()` 让 PyTorch 计算
每个参数的梯度。

本题只计算梯度，不使用优化器，也不修改参数。

场景
====

模型参数与数据固定为：

    weight = 1.0
    bias = 0.0
    observation = 0.5
    target = 1.0

当前预测是 0.5，低于目标 1.0。平方损失是：

    loss = (prediction - target) ** 2

已有接口
========

calculate_loss_and_gradients(model, observation, target) 接收一个已经创建的
`OneInputQModule` 和两个标量张量，必须返回：

    prediction, loss

你的任务
========

只修改 calculate_loss_and_gradients() 中的 TODO：

1. 调用 model(observation) 得到 prediction；
2. 计算预测与目标的平方损失 loss；
3. 对 loss 调用反向计算；
4. 返回 prediction 和 loss。

不要直接给 `.grad` 赋值，不要修改参数，不要修改 check_exercise() 或 main()。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_autograd_q_error

成功条件
========

程序应显示：

    prediction=+0.500 PASS
    loss=+0.250 PASS
    weight_grad=-0.500 PASS
    bias_grad=-1.000 PASS
    model_forward_called=True PASS
    parameters_unchanged=True PASS

最后显示：

    练习通过：backward 已根据损失计算参数梯度

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

梯度只描述损失对参数变化的方向和敏感程度。本题没有学习率、优化器或参数更新，
因此还没有完成一次训练步骤，更不是 DQN 训练。
"""

from __future__ import annotations

import torch

from examples.pytorch_module_prediction import OneInputQModule


def calculate_loss_and_gradients(
    model: OneInputQModule,
    observation: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回预测与损失，并让 PyTorch 填充模型参数的 grad。"""
    predict = model(observation)
    loss = (target - predict) ** 2
    loss.backward()
    return (predict, loss)


def check_exercise() -> bool | None:
    """检查预测、损失、梯度和参数不变性。"""
    model = OneInputQModule(weight=1.0, bias=0.0)
    observation = torch.tensor(0.5, dtype=torch.float32)
    target = torch.tensor(1.0, dtype=torch.float32)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model_forward_called = False

    def remember_forward_call(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal model_forward_called
        model_forward_called = True

    hook = model.register_forward_hook(remember_forward_call)

    try:
        prediction, loss = calculate_loss_and_gradients(
            model, observation, target
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None
    finally:
        hook.remove()

    checks = (
        ("prediction", prediction, 0.5),
        ("loss", loss, 0.25),
        ("weight_grad", model.weight.grad, -0.5),
        ("bias_grad", model.bias.grad, -1.0),
    )
    all_passed = True
    for name, value, expected in checks:
        is_scalar_tensor = isinstance(value, torch.Tensor) and value.ndim == 0
        actual = value.item() if is_scalar_tensor else None
        passed = actual is not None and abs(actual - expected) < 1e-6
        actual_text = f"{actual:+.3f}" if actual is not None else repr(value)
        print(
            f"{name}={actual_text} expected={expected:+.3f} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        all_passed = all_passed and passed

    print(
        f"model_forward_called={model_forward_called} "
        f"{'PASS' if model_forward_called else 'FAIL'}"
    )
    all_passed = all_passed and model_forward_called

    unchanged = all(
        torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )
    print(f"parameters_unchanged={unchanged} {'PASS' if unchanged else 'FAIL'}")
    return all_passed and unchanged


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：backward 已根据损失计算参数梯度")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "calculate_loss_and_gradients() 中的 TODO"
        )


if __name__ == "__main__":
    main()
