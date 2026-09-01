#!/usr/bin/env python3
"""第 38 课编程练习：把两项观察组合成一个预测 Q 值。

为什么有这道题
============

神经网络首先是一个“接收数字、经过计算、返回数字”的模型。PyTorch 会在
以后帮助我们管理大量参数，但本题只使用普通 Python 函数，看清最基本的
输入、参数、局部计算和 return。

场景
====

我们暂时只估计“向右推”这个动作的 Q 值。模型看到两项简化观察：

    pole_angle             杆的角度
    pole_angular_velocity  杆的角速度

每项观察都有一个对应权重。模型需要：

1. 让杆角度乘以角度权重；
2. 让杆角速度乘以角速度权重；
3. 把两个结果相加；
4. 最后加上 bias，并返回预测值。

已有接口
========

predict_right_q(observation, weights, bias) 接收：

    observation = (pole_angle, pole_angular_velocity)
    weights = (angle_weight, angular_velocity_weight)
    bias = 一个额外的可调数字

你的任务
========

只修改 predict_right_q() 中的 TODO。可以先把元组中的数字分别取出，再计算并
return。不要修改 TEST_CASES、check_exercise() 或 main()。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.python_function_q_prediction

成功条件
========

三个场景都打印 PASS，最后显示：

    练习通过：你已经用一个 Python 函数组合两项观察并返回预测值

如果 TODO 尚未完成，程序只显示友好提示，不会输出异常堆栈，也不会影响仓库
全量测试。

当前没有验证
============

本题没有使用 PyTorch，也没有创建真正的多层神经网络、计算预测误差或更新
参数。完成它只证明你理解了 Python 函数怎样使用输入和参数得到一个输出。
"""

from __future__ import annotations

from math import isclose


Observation = tuple[float, float]
Weights = tuple[float, float]


def predict_right_q(
    observation: Observation,
    weights: Weights,
    bias: float,
) -> float:
    """根据两项观察和对应参数返回“向右推”的预测 Q 值。"""
    return bias + observation[0] * weights[0] + observation[1] * weights[1]


TEST_CASES = (
    ((0.1, 0.2), (2.0, -1.0), 0.5, 0.5),
    ((0.5, -0.1), (2.0, -1.0), 0.5, 1.6),
    ((0.0, 0.0), (2.0, -1.0), 0.5, 0.5),
)


def check_exercise() -> bool | None:
    """运行三个可见场景；None 表示练习还没有填写。"""
    all_passed = True
    for index, (observation, weights, bias, expected) in enumerate(
        TEST_CASES, start=1
    ):
        try:
            actual = predict_right_q(observation, weights, bias)
        except NotImplementedError as error:
            print(f"练习尚未完成：{error}")
            return None

        passed = isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        print(
            f"场景{index}: observation={observation} weights={weights} "
            f"bias={bias:+.1f} prediction={actual:+.3f} "
            f"expected={expected:+.3f} {'PASS' if passed else 'FAIL'}"
        )
        all_passed = all_passed and passed
    return all_passed


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：你已经用一个 Python 函数组合两项观察并返回预测值")
    elif result is False:
        print()
        print("还有场景未通过，请只检查 predict_right_q() 中的 TODO")


if __name__ == "__main__":
    main()
