#!/usr/bin/env python3
"""第 60 课编程练习：用一个线性层把 CartPole 四项观察变成两个 Q 值。

为什么有这道题
================

第 59 课得到的 CartPole observation 有四个数，而第 46 课的旧网络只接收两个数。
线性层的输入列数必须和每条观察的特征数一致，否则矩阵乘法无法进行。

固定数据与手算
==============

本题固定一条简化观察：

    observation = [0.2, -0.1, 0.3, 0.0]

四项依次表示小车位置、小车速度、杆角度、杆角速度。两个动作的固定参数为：

    action 0 权重 = [ 1.0, 2.0, -1.0,  0.5]，bias = 0.1
    action 1 权重 = [-1.0, 0.5,  2.0, -0.5]，bias = 0.2

动作 0 的 Q 值：

    1.0×0.2 + 2.0×(-0.1) + (-1.0)×0.3 + 0.5×0.0 + 0.1
    = 0.2 - 0.2 - 0.3 + 0.0 + 0.1
    = -0.2

动作 1 的 Q 值：

    (-1.0)×0.2 + 0.5×(-0.1) + 2.0×0.3 + (-0.5)×0.0 + 0.2
    = -0.2 - 0.05 + 0.6 + 0.0 + 0.2
    = 0.55

所以模型应输出：

    q_values = [-0.2, 0.55]
    best_action = 1

已知 PyTorch 接口
================

类中已经提供属性类型声明：

    q_values: nn.Linear

这行只告诉 VSCode/Pylance：`q_values` 将来一定是可调用的线性层。它不会创建
对象，也不会登记参数；真正的线性层仍必须在 `__init__()` 中赋值给
`self.q_values`。

创建四输入、两输出的线性层：

    self.q_values = nn.Linear(
        in_features=4,
        out_features=2,
    )

它会登记：

    weight.shape = (2, 4)
    bias.shape = (2,)

权重有两行：第 0 行计算动作 0 的 Q 值，第 1 行计算动作 1 的 Q 值。每行有
四列，因为每个动作的分数都要读取四项观察。

在 `forward()` 中调用这一层：

    q_values = self.q_values(observation)
    return q_values

输入一条观察时：

    observation.shape = (4,)
    q_values.shape = (2,)

输入一批两条观察时：

    observations.shape = (2, 4)
    q_values.shape = (2, 2)

线性层自动把同一组参数分别应用到批次中的每一行。

基础任务
========

只修改 `CartPoleLinearQNetwork` 中的两个 TODO：

1. 在 `__init__()` 创建 `nn.Linear(in_features=4, out_features=2)`，保存为
   `self.q_values`；
2. 在 `forward()` 中把 `observation` 交给 `self.q_values` 并返回结果。

不要填写固定权重；检查器会在模型创建后写入它们。不要添加 softmax、ReLU、
隐藏层、loss、backward 或 optimizer，也不要修改带类型的 `__call__()`。

进阶任务：批量预测和逐行动作选择
================================

基础模型通过后，继续完成 `predict_cartpole_batch()`。输入是多条 Python 四元组：

    observations_data = [
        (0.2, -0.1, 0.3, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (-0.2, 0.1, -0.3, 0.0),
    ]

先把整批转换为 `float32` 张量。完整接口是：

    observations = torch.tensor(
        observations_data,
        dtype=torch.float32,
    )

必须在调用模型前检查：批次不能是空列表，张量必须恰好是二维，并且最后一维
必须等于 4。相关属性是：

    observations.ndim
    observations.shape[1]

不符合时抛出：

    raise ValueError("说明原因")

预测阶段不需要构造训练计算图，已知写法是：

    with torch.no_grad():
        # 在这里调用模型并选择动作

模型一次接收整个批次，返回 `(3, 2)`。然后每一行都要在两个动作列之间选择：

    best_actions = q_values.argmax(dim=1)

`dim=1` 表示横跨每一行的动作列。不能写 `dim=0`：那会跨不同经验比较，回答
“哪个样本让某动作分数最高”，而不是“每条经验选择哪个动作”。

这组固定参数的预期输出为：

    q_values = [
        [-0.2,  0.55],
        [ 0.1,  0.2 ],
        [ 0.4, -0.15],
    ]
    best_actions = [1, 1, 0]

只修改 `predict_cartpole_batch()` 中的 TODO：

1. 空列表时在模型调用前抛出 `ValueError`；
2. 转成 `torch.float32` 张量；
3. shape 不是 `(任意正数, 4)` 时在模型调用前抛出 `ValueError`；
4. 在 `torch.no_grad()` 中只调用模型一次；
5. 用 `argmax(dim=1)` 得到每一行的动作索引；
6. 返回 `q_values, best_actions`。

不要用 Python 循环逐条调用模型，不要手写固定 Q 值，不要把三条观察展平成长度
12 的向量，也不要使用 `argmax(dim=0)`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.cartpole_four_input_q_network

成功条件
========

程序会检查线性层登记、参数形状、参数数量、手算 Q 值、批量 Q 值、逐行动作、
模型只调用一次、无梯度预测、非法输入提前拒绝，以及参数不变性。最后应显示：

    练习通过：CartPole 批量观察已逐行映射为 Q 值和动作

TODO 未完成时只显示友好提示，不会输出异常堆栈或破坏全量测试。

当前没有验证
============

本题只验证单层线性前向计算，没有隐藏层或 ReLU，没有根据网络输出探索，没有
loss、参数更新、CartPole 完整训练、检查点、独立评估、Isaac Lab 或真机验证。
"""

from __future__ import annotations

import torch
from torch import nn


class CartPoleLinearQNetwork(nn.Module):
    """接收四项 CartPole 观察，输出 [LEFT, RIGHT] 两个 Q 值。"""

    # 只给 VSCode/Pylance 声明属性类型；实际对象仍由 TODO 1 创建。
    q_values: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        # 1：创建四输入、两输出的 nn.Linear，保存为 self.q_values。
        self.q_values = nn.Linear(4, 2)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """保留 Module 调用流程，并让 VSCode 推断返回 Tensor。"""
        return super().__call__(observation)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """返回顺序为 [LEFT, RIGHT] 的两个 Q 值。"""
        # 2：调用 self.q_values(observation) 并返回结果。
        return self.q_values(observation)


def predict_cartpole_batch(
    model: CartPoleLinearQNetwork,
    observations_data: list[tuple[float, ...]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """校验并批量预测，再为每一行选择最大 Q 值的动作。"""
    # 3：先拒绝空的 observations_data，不能先调用模型。
    if len(observations_data) == 0:
        raise ValueError("数据为空")

    # 4：使用文件顶部给出的完整 torch.tensor 接口，创建
    # dtype=torch.float32 的 observations。
    observations = torch.tensor(observations_data, dtype=torch.float32)

    # 5：检查 observations 必须是二维且 shape[1] == 4，
    # 否则在调用模型前抛出 ValueError。
    if len(observations.shape) != 2:
        raise ValueError("observations不是二维的")
    if observations.shape[1] != 4:
        raise ValueError("观察的数量不对")

    # 6：在 torch.no_grad() 中只调用一次 model(observations)，
    # 得到 q_values；再用 q_values.argmax(dim=1) 得到 best_actions。
    with torch.no_grad():
        q_values = model(observations)
        best_actions = q_values.argmax(dim=1)

    # 7：返回 q_values, best_actions。
        return (q_values, best_actions)

def set_reference_parameters(model: CartPoleLinearQNetwork) -> None:
    """写入与文件顶部手算一致的参数。"""
    with torch.no_grad():
        model.q_values.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, -1.0, 0.5],
                    [-1.0, 0.5, 2.0, -0.5],
                ],
                dtype=torch.float32,
            )
        )
        model.q_values.bias.copy_(
            torch.tensor([0.1, 0.2], dtype=torch.float32)
        )


def check_exercise() -> bool | None:
    """检查层结构、手算输出、批量形状和参数不变性。"""
    try:
        model = CartPoleLinearQNetwork()
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    layer = getattr(model, "q_values", None)
    layer_ok = isinstance(layer, nn.Linear)
    print(
        f"linear_layer_registered={layer_ok} "
        f"{'PASS' if layer_ok else 'FAIL'}"
    )
    if not layer_ok:
        return False

    shapes_ok = (
        tuple(layer.weight.shape) == (2, 4)
        and layer.bias is not None
        and tuple(layer.bias.shape) == (2,)
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    parameter_count_ok = parameter_count == 10

    set_reference_parameters(model)
    before = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    observation = torch.tensor(
        [0.2, -0.1, 0.3, 0.0], dtype=torch.float32
    )
    try:
        q_values = model(observation)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    expected = torch.tensor([-0.2, 0.55], dtype=torch.float32)
    single_shape_ok = tuple(q_values.shape) == (2,)
    values_ok = single_shape_ok and torch.allclose(q_values, expected)
    best_action = q_values.argmax().item() if single_shape_ok else None
    best_action_ok = best_action == 1

    batch = torch.stack(
        (observation, torch.zeros(4, dtype=torch.float32))
    )
    batch_q_values = model(batch)
    batch_ok = (
        tuple(batch_q_values.shape) == (2, 2)
        and torch.allclose(batch_q_values[0], expected)
        and torch.allclose(
            batch_q_values[1],
            torch.tensor([0.1, 0.2], dtype=torch.float32),
        )
    )
    parameters_unchanged = all(
        torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )

    observations_data = [
        (0.2, -0.1, 0.3, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (-0.2, 0.1, -0.3, 0.0),
    ]
    forward_calls = 0

    def count_forward_call(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal forward_calls
        forward_calls += 1

    hook = model.register_forward_hook(count_forward_call)
    try:
        try:
            batch_values, batch_actions = predict_cartpole_batch(
                model=model,
                observations_data=observations_data,
            )
        except NotImplementedError as error:
            print(f"进阶练习尚未完成：{error}")
            return None

        calls_after_valid_batch = forward_calls
        invalid_inputs_rejected_before_model = True
        for invalid_data in (
            [],
            [(0.1, 0.2, 0.3)],
            [(0.1, 0.2, 0.3, 0.4, 0.5)],
        ):
            calls_before_invalid = forward_calls
            try:
                predict_cartpole_batch(
                    model=model,
                    observations_data=invalid_data,
                )
            except ValueError:
                pass
            else:
                invalid_inputs_rejected_before_model = False
            invalid_inputs_rejected_before_model = (
                invalid_inputs_rejected_before_model
                and forward_calls == calls_before_invalid
            )
    finally:
        hook.remove()

    expected_batch_values = torch.tensor(
        [
            [-0.2, 0.55],
            [0.1, 0.2],
            [0.4, -0.15],
        ],
        dtype=torch.float32,
    )
    batch_values_ok = (
        isinstance(batch_values, torch.Tensor)
        and batch_values.shape == (3, 2)
        and batch_values.dtype == torch.float32
        and torch.allclose(batch_values, expected_batch_values)
    )
    batch_actions_ok = (
        isinstance(batch_actions, torch.Tensor)
        and batch_actions.shape == (3,)
        and batch_actions.dtype == torch.long
        and torch.equal(batch_actions, torch.tensor([1, 1, 0]))
    )
    called_model_once = calls_after_valid_batch == 1
    prediction_has_no_grad = (
        isinstance(batch_values, torch.Tensor)
        and not batch_values.requires_grad
    )
    parameters_still_unchanged = all(
        torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )

    shown_values = (
        [round(value, 3) for value in q_values.tolist()]
        if single_shape_ok
        else repr(q_values)
    )
    checks = (
        ("parameter_shapes=(2, 4)/(2,)", shapes_ok),
        ("parameter_count=10", parameter_count_ok),
        (f"q_values={shown_values}", values_ok),
        (f"best_action={best_action}", best_action_ok),
        ("batch_input_shape=(2, 4)_output_shape=(2, 2)", batch_ok),
        ("parameters_unchanged=True", parameters_unchanged),
        ("challenge_batch_q_values_match=True", batch_values_ok),
        ("challenge_best_actions=[1, 1, 0]", batch_actions_ok),
        ("challenge_model_called_once=True", called_model_once),
        ("challenge_prediction_requires_grad=False", prediction_has_no_grad),
        (
            "challenge_invalid_inputs_rejected_before_model=True",
            invalid_inputs_rejected_before_model,
        ),
        (
            "challenge_parameters_still_unchanged=True",
            parameters_still_unchanged,
        ),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：CartPole 批量观察已逐行映射为 Q 值和动作")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "CartPoleLinearQNetwork 和 predict_cartpole_batch 中的 TODO"
        )


if __name__ == "__main__":
    main()
