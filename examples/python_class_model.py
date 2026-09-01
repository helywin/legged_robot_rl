#!/usr/bin/env python3
"""用普通 Python 类保存模型参数，并重复进行预测。"""

from __future__ import annotations


class OneInputQModel:
    """保存一个权重和一个偏置的最小模型对象。"""

    def __init__(self, weight: float, bias: float) -> None:
        self.weight = weight
        self.bias = bias

    def predict(self, observation: float) -> float:
        """使用对象内部保存的参数计算预测值。"""
        return observation * self.weight + self.bias


def run_demo() -> None:
    model = OneInputQModel(weight=3.0, bias=0.1)

    print(f"stored_weight={model.weight:+.2f}")
    print(f"stored_bias={model.bias:+.2f}")
    for observation in (-0.02, 0.05):
        print(
            f"observation={observation:+.2f} "
            f"prediction={model.predict(observation):+.2f}"
        )


if __name__ == "__main__":
    run_demo()
