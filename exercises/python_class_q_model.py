#!/usr/bin/env python3
"""第 39 课编程练习：让 Python 对象保存两项权重和偏置。

为什么有这道题
============

上一课的函数每次调用都要传入 observation、weights 和 bias。真正的模型会把
参数保存在自己内部，预测时只需要接收新的 observation。

本题只学习 Python 类怎样保存并使用参数，不使用 PyTorch，也不更新参数。

场景
====

RightQModel 仍然估计“向右推”的 Q 值。创建模型对象时传入：

    weights = (angle_weight, angular_velocity_weight)
    bias = 一个额外的可调数字

对象需要把它们保存下来。之后调用 model.predict(observation) 时，只传入：

    observation = (pole_angle, pole_angular_velocity)

已有接口
========

    model = RightQModel(weights=(2.0, -1.0), bias=0.5)
    prediction = model.predict(observation=(0.1, 0.2))

你的任务
========

只修改 RightQModel 中的两个 TODO：

1. 在 __init__() 中把 weights 和 bias 保存到当前对象 self；
2. 在 predict() 中读取 observation 和对象保存的参数，返回预测值。

不要修改 TEST_CASES、check_exercise() 或 main()。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.python_class_q_model

成功条件
========

程序会检查：

1. 参数确实保存在对象内部；
2. 三组预测全部正确；
3. 两个模型对象能各自保存不同参数，互不覆盖。

最后应显示：

    练习通过：模型对象能够保存并使用自己的参数

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏全量测试。

当前没有验证
============

完成本题只证明你理解 class、对象和 self 怎样组织参数。它不是 PyTorch 网络，
没有训练、预测误差、反向传播或 DQN。
"""

from __future__ import annotations

from math import isclose


Observation = tuple[float, float]
Weights = tuple[float, float]


class RightQModel:
    """保存右推动作参数的最小模型；只修改两个 TODO。"""

    def __init__(self, weights: Weights, bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def predict(self, observation: Observation) -> float:
        return self.bias + observation[0] * self.weights[0] + observation[1] * self.weights[1]


TEST_CASES = (
    ((0.1, 0.2), 0.5),
    ((0.5, -0.1), 1.6),
    ((0.0, 0.0), 0.5),
)


def check_exercise() -> bool | None:
    """检查参数保存、预测和对象独立性。"""
    try:
        model = RightQModel(weights=(2.0, -1.0), bias=0.5)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    stored = getattr(model, "weights", None) == (2.0, -1.0) and isclose(
        getattr(model, "bias", float("nan")), 0.5
    )
    print(f"参数保存在对象内部：{'PASS' if stored else 'FAIL'}")

    all_passed = stored
    for index, (observation, expected) in enumerate(TEST_CASES, start=1):
        try:
            actual = model.predict(observation)
        except NotImplementedError as error:
            print(f"练习尚未完成：{error}")
            return None
        except AttributeError as error:
            print(f"对象中缺少需要的参数：{error}")
            return False

        passed = isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        print(
            f"场景{index}: observation={observation} "
            f"prediction={actual:+.3f} expected={expected:+.3f} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        all_passed = all_passed and passed

    try:
        other_model = RightQModel(weights=(-2.0, 1.0), bias=-0.5)
        first_value = model.predict((0.2, 0.1))
        other_value = other_model.predict((0.2, 0.1))
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    independent = isclose(first_value, 0.8) and isclose(other_value, -0.8)
    print(f"两个对象参数互不覆盖：{'PASS' if independent else 'FAIL'}")
    return all_passed and independent


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：模型对象能够保存并使用自己的参数")
    elif result is False:
        print()
        print("还有检查未通过，请只修改 RightQModel 中的两个 TODO")


if __name__ == "__main__":
    main()
