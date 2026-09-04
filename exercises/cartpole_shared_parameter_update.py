#!/usr/bin/env python3
"""第 63 课可运行训练：一条训练数据怎样改变多个预测。

这次不再让你填写一大组快照，也不再把教学检查混进训练函数。
文件已经给出完整答案，你只需要运行并观察输出。

本文件里的两个 observation 职责不同：

    training_observation = [0.2, -0.1, 0.3, 0.0]
        真正参加训练。它经过前向计算，产生 loss，随后触发 backward 和 step。

    comparison_observation = [0.0, 0.0, 0.0, 0.0]
        只是一把“测量尺”。它在更新前、更新后各预测一次，不产生 loss，
        不参加 backward，也不会单独训练模型。

训练主线只有下面七步：

    清旧梯度
    → training_observation 前向
    → 选出实际动作的 Q 值
    → 用 target 算 loss
    → backward() 把梯度写入 parameter.grad
    → optimizer.step() 读取 grad 并修改参数
    → 返回本次更新前的 loss

固定数据：

    action_index = 1
    target_q = 0.7
    learning_rate = 0.1

预期现象：

    训练观察 A：Q 从 [0.1, 0.3] 变成 [-0.1736, 0.8438592]
    对照观察 B：Q 从 [0.0, 0.3] 变成 [-0.16, 0.70416]
    A 的 loss：从 0.16 降到约 0.02069547

B 没有参与训练。它的预测会变化，是因为 A 的 loss 修改了两条观察共用的
网络参数；B 在更新后重新前向时读取了这些新参数。

运行命令：

    .venv/bin/python -m exercises.cartpole_shared_parameter_update

本文件仍只验证一条人为 target 的一次 SGD 更新，不是完整 DQN 训练，也没有
验证 Gymnasium、Isaac Lab 或真机。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from exercises.cartpole_relu_hidden_q_network import (
    CartPoleReluQNetwork,
    set_reference_parameters,
)


def train_on_one_observation(
    model: CartPoleReluQNetwork,
    training_observation: torch.Tensor,
    action_index: int,
    target_q: float,
    learning_rate: float,
) -> torch.Tensor:
    """只用 training_observation 完成一次 SGD 参数更新。

    返回值是更新发生前、由旧参数算出的 loss 数值快照。函数不接收
    comparison_observation，因为对照观察不属于训练算法。
    """
    if action_index not in (0, 1):
        raise ValueError("action_index 必须是 0 或 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")

    # optimizer 保存 model 参数对象的引用，不保存也不绑定 loss。
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    # 第 1 步：清除参数上一次训练留下的梯度；参数数值此时没有改变。
    optimizer.zero_grad(set_to_none=True)

    # 第 2 步：用旧参数预测训练观察的两个动作 Q 值，并建立计算图。
    all_action_q_values = model(training_observation)

    # 第 3 步：经验里实际执行的是 action_index，只把该动作接入 loss。
    executed_action_q = all_action_q_values[action_index]

    # 第 4 步：target 是比较标准，不需要追踪梯度。
    target_q_tensor = torch.tensor(
        target_q,
        dtype=executed_action_q.dtype,
        device=executed_action_q.device,
    )
    loss = F.mse_loss(executed_action_q, target_q_tensor)

    # 第 5 步：loss 沿本轮计算图反传，把梯度写入各参数的 .grad。
    loss.backward()

    # 第 6 步：optimizer 读取它管理的参数及其 .grad，原地修改参数。
    optimizer.step()

    # 返回独立数值快照；训练主线不承担前后对照与教学检查。
    return loss.detach().clone()


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    """保存不会随模型后续更新而变化的数值副本。"""
    return tensor.detach().clone()


def _required_bias(layer: nn.Linear, label: str) -> torch.Tensor:
    """取得本课固定网络必有的 bias，并为 Pylance 收窄类型。"""
    bias = layer.bias
    if bias is None:
        raise RuntimeError(f"{label} 不应为 None")
    return bias


def _required_gradient(tensor: torch.Tensor, label: str) -> torch.Tensor:
    """取得 backward 写入的梯度，并为 Pylance 收窄类型。"""
    gradient = tensor.grad
    if gradient is None:
        raise RuntimeError(f"{label}.grad 仍是 None")
    return gradient


def _allclose(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return actual.shape == expected.shape and torch.allclose(
        actual,
        expected,
        atol=1e-6,
    )


def _format_q_values(q_values: torch.Tensor) -> str:
    """只格式化打印结果，不改变参与训练和检查的原始张量。"""
    values = (f"{value:.4f}" for value in q_values.tolist())
    return f"[{', '.join(values)}]"


def check_exercise() -> bool:
    """在训练函数外完成前后测量，证明共享参数造成预测联动。"""
    model = CartPoleReluQNetwork()
    set_reference_parameters(model)

    # A 是训练材料：只有它会进入 train_on_one_observation()。
    training_observation = torch.tensor(
        [0.2, -0.1, 0.3, 0.0],
        dtype=torch.float32,
    )

    # B 是测量尺：不传入训练函数，不产生 loss，不参与 backward。
    comparison_observation = torch.tensor(
        [0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    # 测量阶段 1：更新前只读地记录 A、B 的预测和模型参数。
    with torch.no_grad():
        training_q_before = _snapshot(model(training_observation))
        comparison_q_before = _snapshot(model(comparison_observation))
        hidden_weight_before = _snapshot(model.hidden_layer.weight)
        hidden_bias_before = _snapshot(
            _required_bias(model.hidden_layer, "hidden_layer.bias")
        )
        output_weight_before = _snapshot(model.output_layer.weight)
        output_bias_before = _snapshot(
            _required_bias(model.output_layer, "output_layer.bias")
        )

    # 故意塞入 99，检查训练函数是否真的在本轮开始前清除了旧梯度。
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 99.0)

    # 训练阶段：只有 A 进入这次前向、loss、backward 和 step。
    loss_before = train_on_one_observation(
        model=model,
        training_observation=training_observation,
        action_index=1,
        target_q=0.7,
        learning_rate=0.1,
    )

    # backward 写入的梯度在 step 后仍可查看；step 读取它但不会自动清空它。
    output_weight_gradient = _snapshot(
        _required_gradient(model.output_layer.weight, "output_layer.weight")
    )

    # 测量阶段 2：用更新后的同一个 model，再只读地预测 A 和 B。
    with torch.no_grad():
        training_q_after = _snapshot(model(training_observation))
        comparison_q_after = _snapshot(model(comparison_observation))
        hidden_weight_after = _snapshot(model.hidden_layer.weight)
        hidden_bias_after = _snapshot(
            _required_bias(model.hidden_layer, "hidden_layer.bias")
        )
        output_weight_after = _snapshot(model.output_layer.weight)
        output_bias_after = _snapshot(
            _required_bias(model.output_layer, "output_layer.bias")
        )
        target = torch.tensor(0.7, dtype=training_q_after.dtype)
        loss_after = F.mse_loss(training_q_after[1], target)

    print("阶段 1：更新前，只测量，不训练")
    print(f"  训练观察 A 的 Q 值：{_format_q_values(training_q_before)}")
    print(f"  对照观察 B 的 Q 值：{_format_q_values(comparison_q_before)}")
    print("阶段 2：只让观察 A 产生 loss → backward → optimizer.step")
    print(f"  本次 loss：{loss_before.item():.8f}")
    print("阶段 3：更新后，仍然只测量")
    print(f"  训练观察 A 的 Q 值：{_format_q_values(training_q_after)}")
    print(f"  对照观察 B 的 Q 值：{_format_q_values(comparison_q_after)}")
    print(f"  用新参数重算 A 的 loss：{loss_after.item():.8f}")

    expected_hidden_weight_after = torch.tensor(
        [
            [0.984, 0.008, -0.024, 0.0],
            [0.0, 2.0, -1.0, 0.0],
            [-0.968, -0.016, 1.048, 0.0],
        ]
    )
    expected_hidden_bias_after = torch.tensor([-0.08, 0.0, 0.26])
    expected_output_weight_after = torch.tensor(
        [[1.0, 2.0, -1.0], [-0.984, 0.5, 2.016]]
    )
    expected_output_bias_after = torch.tensor([0.1, 0.18])

    stale_gradient_cleared = not torch.any(
        output_weight_gradient == 99.0
    ).item()
    selected_action_gradient_ok = _allclose(
        output_weight_gradient,
        torch.tensor([[0.0, 0.0, 0.0], [-0.16, 0.0, -0.16]]),
    )
    parameter_update_ok = (
        _allclose(hidden_weight_after, expected_hidden_weight_after)
        and _allclose(hidden_bias_after, expected_hidden_bias_after)
        and _allclose(output_weight_after, expected_output_weight_after)
        and _allclose(output_bias_after, expected_output_bias_after)
    )
    unchanged_paths_ok = (
        torch.equal(hidden_weight_before[1], hidden_weight_after[1])
        and hidden_bias_before[1].item() == hidden_bias_after[1].item()
        and torch.equal(output_weight_before[0], output_weight_after[0])
        and output_bias_before[0].item() == output_bias_after[0].item()
    )
    training_prediction_ok = (
        _allclose(training_q_before, torch.tensor([0.1, 0.3]))
        and _allclose(
            training_q_after,
            torch.tensor([-0.1736, 0.8438592]),
        )
        and _allclose(loss_before, torch.tensor(0.16))
        and abs(loss_after.item() - 0.02069547) < 1e-6
        and loss_after.item() < loss_before.item()
    )
    comparison_prediction_ok = (
        _allclose(comparison_q_before, torch.tensor([0.0, 0.3]))
        and _allclose(
            comparison_q_after,
            torch.tensor([-0.16, 0.70416]),
        )
        and not torch.equal(comparison_q_before, comparison_q_after)
    )

    checks = (
        ("旧梯度已清除", stale_gradient_cleared),
        ("实际动作的梯度路径正确", selected_action_gradient_ok),
        ("SGD 参数更新符合公式", parameter_update_ok),
        ("零梯度对应的参数保持不变", unchanged_paths_ok),
        ("训练观察的 loss 下降", training_prediction_ok),
        ("未训练的对照观察也读取了新参数", comparison_prediction_ok),
    )
    print("检查结果：")
    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}：{label}")

    passed = all(result for _, result in checks)
    if passed:
        print("\n运行通过：训练主线与共享参数造成的预测联动已经可见")
    else:
        print("\n运行未通过：请把 FAIL 项和上面的实际数值发给我")
    return passed


if __name__ == "__main__":
    if not check_exercise():
        raise SystemExit(1)
