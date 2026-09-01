#!/usr/bin/env python3
"""第 45 课编程练习：把一次参数更新重复成最小训练循环。

为什么有这道题
============

前四课已经分别完成模型预测、计算梯度、更新参数和清空旧梯度。本题把这些步骤放进
`for` 循环，让同一个模型针对同一条固定数据连续学习八步。

本题只学习训练循环的顺序，不引入批量数据、环境交互或 DQN。

场景
====

模型与数据固定为：

    weight = 1.0
    bias = 0.0
    observation = 0.5
    target = 1.0
    learning_rate = 0.1
    steps = 8

初始预测是 `0.5`，目标是 `1.0`。每一步都应使用更新后的同一个模型继续预测，
使损失持续下降。

已有接口
========

`train_for_steps(model, optimizer, observation, target, steps)` 接收模型、优化器、
一条数据和训练步数，返回一个 Python 列表：

    losses

列表按顺序保存每一步参数更新前的损失数值。

你的任务
========

只修改 `train_for_steps()` 中的 TODO，完成整个 `for` 循环：

1. 每一步先清空旧梯度；
2. 调用模型得到预测；
3. 计算平方损失；
4. 反向计算梯度；
5. 让优化器更新参数；
6. 把本步损失的普通 Python 数值追加到 losses；
7. 循环结束后返回 losses。

不要重新创建模型或优化器，不要直接修改参数或 `.grad`，不要修改
`check_exercise()` 或 `main()`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_training_loop

成功条件
========

程序会检查：

    recorded_steps=8 PASS
    first_loss=0.250000 PASS
    loss_kept_decreasing=True PASS
    final_prediction=0.949944 PASS
    final_loss=0.002506 PASS

最后显示：

    练习通过：单次更新已组成八步训练循环

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

这里只反复使用同一条固定数据，不能证明模型能处理多条数据、CartPole 环境或 DQN。
"""

from __future__ import annotations

import torch

from examples.pytorch_module_prediction import OneInputQModule


def train_for_steps(
    model: OneInputQModule,
    optimizer: torch.optim.Optimizer,
    observation: torch.Tensor,
    target: torch.Tensor,
    steps: int,
) -> list[float]:
    """重复训练 steps 步，并返回每一步更新前的损失。"""
    losses = list()
    for i in range(0, steps):
        optimizer.zero_grad()
        predict = model(observation)
        loss = (target - predict) ** 2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def check_exercise() -> bool | None:
    """检查训练步数、损失变化和最终预测。"""
    model = OneInputQModule(weight=1.0, bias=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    observation = torch.tensor(0.5, dtype=torch.float32)
    target = torch.tensor(1.0, dtype=torch.float32)

    try:
        losses = train_for_steps(
            model, optimizer, observation, target, steps=8
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    recorded_steps = isinstance(losses, list) and len(losses) == 8
    values_are_floats = recorded_steps and all(
        isinstance(loss, float) for loss in losses
    )
    first_loss_ok = values_are_floats and abs(losses[0] - 0.25) < 1e-6
    kept_decreasing = values_are_floats and all(
        earlier > later for earlier, later in zip(losses, losses[1:])
    )

    with torch.no_grad():
        final_prediction = model(observation).item()
        final_loss = (final_prediction - target.item()) ** 2
    final_prediction_ok = abs(final_prediction - 0.94994366) < 1e-6
    final_loss_ok = abs(final_loss - 0.00250564) < 1e-6

    checks = (
        ("recorded_steps=8", recorded_steps),
        ("loss_values_are_float", values_are_floats),
        ("first_loss=0.250000", first_loss_ok),
        ("loss_kept_decreasing=True", kept_decreasing),
        (f"final_prediction={final_prediction:.6f}", final_prediction_ok),
        (f"final_loss={final_loss:.6f}", final_loss_ok),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：单次更新已组成八步训练循环")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "train_for_steps() 中的 TODO"
        )


if __name__ == "__main__":
    main()
