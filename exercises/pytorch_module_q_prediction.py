#!/usr/bin/env python3
"""第 41 课编程练习：用 nn.Module 登记参数并完成两项观察预测。

为什么有这道题
============

前面已经学会普通 Python 类保存参数，也学会张量计算。`nn.Module` 把这两件事
组织成 PyTorch 的标准模型形式，使框架能够找到模型中的可学习参数。

本题只学习 `nn.Module`、`nn.Parameter` 和 `forward()` 的职责，不更新参数。

场景
====

RightQModule 仍然估计“向右推”的 Q 值。创建模块时接收两个已经存在的张量：

    weights.shape == (2,)
    bias.ndim == 0

模块要把它们登记成 `nn.Parameter`。调用 `model(observation)` 时，PyTorch 会
自动执行 `forward(observation)`，返回标量预测张量。

已有接口
========

    model = RightQModule(
        weights=torch.tensor([2.0, -1.0]),
        bias=torch.tensor(0.5),
    )
    prediction = model(torch.tensor([0.1, 0.2]))

你的任务
========

只修改 RightQModule 中的两个 TODO：

1. 在 `__init__()` 中先初始化 `nn.Module`，再把 weights 和 bias 登记为对象参数；
2. 在 `forward()` 中使用 observation 和对象参数返回预测值。

建议复制传入张量后再登记，避免模型参数与外部张量共享同一份存储。不要修改
TEST_CASES、check_exercise() 或 main()。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_module_q_prediction

成功条件
========

程序会检查：

1. 模型确实是 `nn.Module`；
2. `weights` 和 `bias` 能从 `named_parameters()` 找到；
3. 三组 `model(observation)` 预测正确；
4. 只做预测不会修改参数。

最后应显示：

    练习通过：nn.Module 已登记参数并通过 forward 完成预测

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

本题没有优化器、损失、反向传播或参数更新。完成它只证明你理解 PyTorch 模型
怎样组织参数和前向计算，还不是训练完成的神经网络或 DQN。
"""

from __future__ import annotations

import torch
from torch import nn


class RightQModule(nn.Module):
    """登记两项权重和偏置；只修改两个 TODO。"""

    def __init__(self, weights: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.bias + observation.dot(self.weights)


TEST_CASES = (
    ([0.1, 0.2], 0.5),
    ([0.5, -0.1], 1.6),
    ([0.0, 0.0], 0.5),
)


def check_exercise() -> bool | None:
    """检查模块类型、参数登记、预测和参数不变性。"""
    try:
        model = RightQModule(
            weights=torch.tensor([2.0, -1.0], dtype=torch.float32),
            bias=torch.tensor(0.5, dtype=torch.float32),
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None
    except AttributeError as error:
        print(f"模块初始化不完整：{error}")
        return False

    is_module = isinstance(model, nn.Module)
    parameters = dict(model.named_parameters())
    registered = set(parameters) == {"weights", "bias"}
    print(f"模型是 nn.Module：{'PASS' if is_module else 'FAIL'}")
    print(f"参数已登记：{'PASS' if registered else 'FAIL'}")

    before = [parameter.detach().clone() for parameter in model.parameters()]
    all_passed = is_module and registered
    for index, (observation_values, expected) in enumerate(TEST_CASES, start=1):
        observation = torch.tensor(observation_values, dtype=torch.float32)
        try:
            prediction = model(observation)
        except NotImplementedError as error:
            print(f"练习尚未完成：{error}")
            return None

        is_scalar_tensor = isinstance(prediction, torch.Tensor) and prediction.ndim == 0
        value_passed = is_scalar_tensor and torch.isclose(
            prediction, torch.tensor(expected, dtype=torch.float32)
        ).item()
        passed = is_scalar_tensor and value_passed
        value_text = (
            f"{prediction.item():+.3f}" if is_scalar_tensor else repr(prediction)
        )
        print(
            f"场景{index}: prediction={value_text} expected={expected:+.3f} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        all_passed = all_passed and passed

    unchanged = all(
        torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )
    print(f"前向预测没有修改参数：{'PASS' if unchanged else 'FAIL'}")
    return all_passed and unchanged


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：nn.Module 已登记参数并通过 forward 完成预测")
    elif result is False:
        print()
        print("还有检查未通过，请只修改 RightQModule 中的两个 TODO")


if __name__ == "__main__":
    main()
