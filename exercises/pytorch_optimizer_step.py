#!/usr/bin/env python3
"""第 43 课编程练习：让优化器根据梯度更新一次模型参数。

为什么有这道题
============

上一课的 `backward()` 已经把梯度写入参数的 `.grad`，但参数本身没有变化。本题
使用 PyTorch 的 SGD 优化器读取这些梯度，再完成一次参数更新。

本题只更新一次，不编写训练循环，也暂不处理多轮梯度清空。

场景
====

模型和数据固定为：

    weight = 1.0
    bias = 0.0
    observation = 0.5
    target = 1.0
    learning_rate = 0.1

更新前预测为 0.5，平方损失为 0.25。`backward()` 会得到：

    weight.grad = -0.5
    bias.grad = -1.0

已有接口
========

`update_parameters_once(model, optimizer, observation, target)` 接收已经创建好的模型
和优化器，必须返回更新前的：

    prediction, loss

你的任务
========

只修改 `update_parameters_once()` 中的 TODO：

1. 调用模型得到更新前预测；
2. 计算预测与目标的平方损失；
3. 调用反向计算，让参数得到梯度；
4. 让传入的 optimizer 使用梯度更新一次参数；
5. 返回更新前的 prediction 和 loss。

不要直接给参数或 `.grad` 赋值，不要创建另一个优化器，不要修改
`check_exercise()` 或 `main()`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_optimizer_step

成功条件
========

程序应显示：

    prediction_before=+0.500 PASS
    loss_before=+0.250 PASS
    weight_after=+1.050 PASS
    bias_after=+0.100 PASS
    prediction_after=+0.625 PASS
    loss_decreased=True PASS

最后显示：

    练习通过：优化器已使用梯度完成一次参数更新

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

本题只更新一次。多轮训练前为什么要清空旧梯度、怎样组织训练循环、怎样训练 DQN，
都还没有验证。
"""

from __future__ import annotations

import torch

from examples.pytorch_module_prediction import OneInputQModule


def update_parameters_once(
    model: OneInputQModule,
    optimizer: torch.optim.Optimizer,
    observation: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predict = model(observation)
    loss = (target - predict) ** 2
    loss.backward()
    optimizer.step()
    return (predict, loss)


def check_exercise() -> bool | None:
    """检查一次 SGD 更新前后的预测、参数和损失。"""
    model = OneInputQModule(weight=1.0, bias=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    observation = torch.tensor(0.5, dtype=torch.float32)
    target = torch.tensor(1.0, dtype=torch.float32)

    try:
        prediction_before, loss_before = update_parameters_once(
            model, optimizer, observation, target
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    with torch.no_grad():
        prediction_after = model(observation)
        loss_after = (prediction_after - target) ** 2

    checks = (
        ("prediction_before", prediction_before, 0.5),
        ("loss_before", loss_before, 0.25),
        ("weight_after", model.weight, 1.05),
        ("bias_after", model.bias, 0.1),
        ("prediction_after", prediction_after, 0.625),
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

    loss_decreased = (
        isinstance(loss_before, torch.Tensor)
        and loss_before.ndim == 0
        and loss_after.item() < loss_before.item()
    )
    print(
        f"loss_decreased={loss_decreased} "
        f"{'PASS' if loss_decreased else 'FAIL'}"
    )
    return all_passed and loss_decreased


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：优化器已使用梯度完成一次参数更新")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "update_parameters_once() 中的 TODO"
        )


if __name__ == "__main__":
    main()
