#!/usr/bin/env python3
"""第 44 课编程练习：第二次更新前清空第一次留下的梯度。

为什么有这道题
============

PyTorch 的 `backward()` 默认把新梯度加到已有 `.grad` 上，而不是覆盖旧值。如果
下一条数据代表不同的调整方向，旧梯度会干扰新的更新。

本题只学习 `optimizer.zero_grad()` 应放在第二次预测和反向计算之前，不编写循环。

场景
====

检查器已经用第一个目标 `1.0` 完成一次更新，模型现在是：

    weight = 1.05
    bias = 0.10

但第一次梯度仍留在 `.grad` 中：

    weight.grad = -0.5
    bias.grad = -1.0

第二个目标改为 `0.4`。当前预测为 `0.625`，这次应该把预测向下调整。如果旧梯度
没有清空，新旧方向相加后会让预测反而升高。

已有接口
========

`update_second_target(model, optimizer, observation, target)` 接收带有旧梯度的模型
和原优化器，后面的预测、损失、`backward()`、`step()` 与返回值已经写好。

你的任务
========

只修改 `update_second_target()` 中的 TODO：

1. 在处理第二个目标前，让传入的 optimizer 清空旧梯度；
2. 不要修改后面的预测、损失、反向计算、更新或返回值。

不要直接给 `.grad` 赋值，不要创建新模型或新优化器，不要修改 `check_exercise()`
或 `main()`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_zero_grad_before_update

成功条件
========

程序应显示：

    prediction_before=+0.625 PASS
    loss_before=+0.051 PASS
    weight_gradient=+0.225 PASS
    bias_gradient=+0.450 PASS
    prediction_after=+0.569 PASS
    moved_toward_target=True PASS

最后显示：

    练习通过：第二次 backward 前已清空旧梯度

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

本题只处理两个固定目标，没有训练循环、批量数据、CartPole DQN 或独立评估。
"""

from __future__ import annotations

import torch

from examples.pytorch_module_prediction import OneInputQModule
from examples.pytorch_zero_grad_demo import prepare_first_update


def update_second_target(
    model: OneInputQModule,
    optimizer: torch.optim.Optimizer,
    observation: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """清空旧梯度，再用第二个目标完成一次更新。"""

    optimizer.zero_grad()
    prediction_before = model(observation)
    loss_before = (prediction_before - target) ** 2
    loss_before.backward()
    optimizer.step()
    return prediction_before, loss_before


def check_exercise() -> bool | None:
    """检查第二次更新只使用第二个目标产生的梯度。"""
    model, optimizer = prepare_first_update()
    observation = torch.tensor(0.5, dtype=torch.float32)
    target = torch.tensor(0.4, dtype=torch.float32)

    try:
        prediction_before, loss_before = update_second_target(
            model, optimizer, observation, target
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    with torch.no_grad():
        prediction_after = model(observation)

    checks = (
        ("prediction_before", prediction_before, 0.625),
        ("loss_before", loss_before, 0.050625),
        ("weight_gradient", model.weight.grad, 0.225),
        ("bias_gradient", model.bias.grad, 0.45),
        ("prediction_after", prediction_after, 0.56875),
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

    moved_toward_target = (
        abs(prediction_after.item() - target.item())
        < abs(prediction_before.item() - target.item())
    )
    print(
        f"moved_toward_target={moved_toward_target} "
        f"{'PASS' if moved_toward_target else 'FAIL'}"
    )
    return all_passed and moved_toward_target


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：第二次 backward 前已清空旧梯度")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "update_second_target() 中的 TODO"
        )


if __name__ == "__main__":
    main()
