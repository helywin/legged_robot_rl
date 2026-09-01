#!/usr/bin/env python3
"""第 40 课编程练习：用 PyTorch 张量组合两项观察。

为什么有这道题
============

前两课已经用普通 Python 函数和类完成了预测。PyTorch 不会改变预测的含义，
但会把数字放进 tensor（张量），以便以后统一进行大量计算和参数更新。

本题只学习张量的创建、逐项乘法和求和，不创建 nn.Module，也不训练参数。

场景
====

仍然估计“向右推”的 Q 值。观察和权重各有两项：

    observation = [pole_angle, pole_angular_velocity]
    weights = [angle_weight, angular_velocity_weight]

张量的 `observation * weights` 会逐项相乘，得到两项观察贡献。随后对贡献调用
`.sum()`，再加上 bias，得到一个标量预测张量。

已有接口
========

predict_right_q(observation, weights, bias) 接收三个已经创建好的张量：

    observation.shape == (2,)
    weights.shape == (2,)
    bias.ndim == 0

你的任务
========

只修改 predict_right_q() 中的 TODO：

1. 使用张量逐项乘法计算两项贡献；
2. 对贡献求和；
3. 加上 bias；
4. 返回标量张量。

不要把张量转换成 Python 列表或浮点数，不要修改 TEST_CASES、check_exercise()
或 main()。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_tensor_q_prediction

成功条件
========

程序会检查返回值仍是零维 PyTorch 张量、三组预测正确，并确认输入张量没有被
修改。最后应显示：

    练习通过：你已经用 PyTorch 张量完成两项观察的预测

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

完成本题只证明你理解张量怎样承载并计算数字。它还不是 nn.Module、多层神经
网络、反向传播、参数更新或 DQN。
"""

from __future__ import annotations

import torch


def predict_right_q(
    observation: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """使用长度为 2 的观察和权重张量返回标量 Q 值。"""
    return bias + observation.dot(weights)

TEST_CASES = (
    ([0.1, 0.2], [2.0, -1.0], 0.5, 0.5),
    ([0.5, -0.1], [2.0, -1.0], 0.5, 1.6),
    ([0.0, 0.0], [2.0, -1.0], 0.5, 0.5),
)


def check_exercise() -> bool | None:
    """检查张量类型、维度、数值和输入不变性。"""
    all_passed = True
    for index, (observation_values, weight_values, bias_value, expected) in enumerate(
        TEST_CASES, start=1
    ):
        observation = torch.tensor(observation_values, dtype=torch.float32)
        weights = torch.tensor(weight_values, dtype=torch.float32)
        bias = torch.tensor(bias_value, dtype=torch.float32)
        before = (observation.clone(), weights.clone(), bias.clone())

        try:
            prediction = predict_right_q(observation, weights, bias)
        except NotImplementedError as error:
            print(f"练习尚未完成：{error}")
            return None

        is_scalar_tensor = isinstance(prediction, torch.Tensor) and prediction.ndim == 0
        value_passed = is_scalar_tensor and torch.isclose(
            prediction, torch.tensor(expected, dtype=torch.float32)
        ).item()
        inputs_unchanged = (
            torch.equal(observation, before[0])
            and torch.equal(weights, before[1])
            and torch.equal(bias, before[2])
        )
        passed = is_scalar_tensor and value_passed and inputs_unchanged

        value_text = (
            f"{prediction.item():+.3f}" if is_scalar_tensor else repr(prediction)
        )
        print(
            f"场景{index}: prediction={value_text} expected={expected:+.3f} "
            f"scalar_tensor={is_scalar_tensor} "
            f"inputs_unchanged={inputs_unchanged} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        all_passed = all_passed and passed
    return all_passed


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：你已经用 PyTorch 张量完成两项观察的预测")
    elif result is False:
        print()
        print("还有检查未通过，请只修改 predict_right_q() 中的 TODO")


if __name__ == "__main__":
    main()
