#!/usr/bin/env python3
"""第 51 课编程练习：把一批逐条误差合成一次在线参数更新。

为什么有这道题
============

经验回放抽出多条经验后，每条经验都有自己的 selected Q 和 target。训练时先分别计算
平方损失，再取平均形成一个标量 loss。`backward()` 会把整批经验的平均贡献写入共享
参数的 `.grad`，optimizer 最后只更新一次。

本题直接提供无梯度 target，只学习批量损失和一次参数更新，不重复目标网络计算。

固定数据
========

    observations = [
        [ 0.2, -0.1],
        [-0.3,  0.4],
    ]
    executed_actions = [1, 0]
    target_q_values = [1.01, 0.20]

更新前网络输出与逐行选值：

    q_values = [
        [0.10, -0.05],
        [0.60,  0.70],
    ]
    selected_q_values = [-0.05, 0.60]

逐条平方损失与平均值：

    per_item_losses = [1.1236, 0.1600]
    loss = mean(per_item_losses) = 0.6418

使用学习率 0.1 更新一次后：

    selected_q_after = [0.0613, 0.55]
    loss_after = 0.5113

你的任务
========

只修改 `batch_loss_update()` 中的 TODO：

1. 清空 optimizer 管理参数的旧梯度；
2. 让 observations 整批进入 model，得到 q_values；
3. 把 executed_actions 变成 `(批量, 1)`，沿动作维逐行 gather，再只压缩动作列维；
4. 分别计算每条经验 `(selected_q_values - target_q_values) ** 2`；
5. 对逐条损失调用 `mean()`，得到标量 loss；
6. 对 loss 反向计算；
7. 让 optimizer 更新一次参数；
8. 返回 q_values、selected_q_values、per_item_losses 和 loss。

不要逐条调用模型或 optimizer，不要使用 `sum()` 代替 `mean()`，不要重新计算或修改
target，不要把任何中间张量转换成 Python 数值，也不要修改检查器或 main。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.pytorch_batch_loss_update

成功条件
========

程序应显示：

    selected_q_before=[-0.05, 0.6] PASS
    per_item_losses=[1.1236, 0.16] PASS
    mean_loss=0.6418 PASS
    loss_is_scalar=True PASS
    both_predictions_moved_toward_targets=True PASS
    selected_q_after=[0.0613, 0.55] PASS
    loss_after=0.5113 PASS
    parameters_changed=True PASS

最后显示：

    练习通过：一批经验已通过平均损失完成一次参数更新

TODO 未完成时只显示友好提示，不会输出异常堆栈，也不会破坏仓库全量测试。

当前没有验证
============

本题没有批量 target 网络计算、经验回放随机抽样、多层网络、完整 CartPole 训练、
检查点或独立评估。
"""

from __future__ import annotations

import torch

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


def batch_loss_update(
    model: TwoActionQModule,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    executed_actions: torch.Tensor,
    target_q_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """用一批固定 target 更新模型一次，并返回更新前关键张量。"""
    # 只修改这里，完成批量选值、逐条损失、平均、反传和更新。
    optimizer.zero_grad()
    q_values = model(observations)
    actions_index = executed_actions.unsqueeze(1)
    selected_q_values_vec = q_values.gather(dim=1, index=actions_index)
    selected_q_values = selected_q_values_vec.squeeze(1)
    loss = (target_q_values - selected_q_values) ** 2
    mean_loss = torch.mean(loss)
    mean_loss.backward()
    optimizer.step()
    return (q_values, selected_q_values, loss, mean_loss)
    


def check_exercise() -> bool | None:
    """检查一批经验是否共同形成一次正确的平均损失更新。"""
    model = TwoActionQModule()
    set_demo_parameters(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    observations = torch.tensor(
        [[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float32
    )
    executed_actions = torch.tensor([1, 0], dtype=torch.long)
    target_q_values = torch.tensor([1.01, 0.20], dtype=torch.float32)
    parameters_before = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    try:
        q_values, selected_q, per_item_losses, loss = batch_loss_update(
            model,
            optimizer,
            observations,
            executed_actions,
            target_q_values,
        )
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        return None

    with torch.no_grad():
        q_values_after = model(observations)
        selected_after = q_values_after.gather(
            1, executed_actions.unsqueeze(1)
        ).squeeze(1)
        per_item_after = (selected_after - target_q_values) ** 2
        loss_after = per_item_after.mean()

    q_values_ok = (
        isinstance(q_values, torch.Tensor)
        and q_values.shape == (2, 2)
        and torch.allclose(
            q_values,
            torch.tensor([[0.1, -0.05], [0.6, 0.7]]),
        )
    )
    selected_ok = (
        isinstance(selected_q, torch.Tensor)
        and selected_q.shape == (2,)
        and torch.allclose(selected_q, torch.tensor([-0.05, 0.6]))
    )
    per_item_ok = (
        isinstance(per_item_losses, torch.Tensor)
        and per_item_losses.shape == (2,)
        and torch.allclose(
            per_item_losses, torch.tensor([1.1236, 0.16])
        )
    )
    loss_ok = (
        isinstance(loss, torch.Tensor)
        and loss.ndim == 0
        and abs(loss.item() - 0.6418) < 1e-6
    )
    moved_toward = selected_ok and all(
        abs(after.item() - target.item())
        < abs(before.item() - target.item())
        for before, after, target in zip(
            selected_q, selected_after, target_q_values
        )
    )
    selected_after_ok = torch.allclose(
        selected_after, torch.tensor([0.0613, 0.55]), atol=1e-6
    )
    loss_after_ok = abs(loss_after.item() - 0.5112659) < 1e-6
    parameters_changed = any(
        not torch.equal(old, current)
        for old, current in zip(parameters_before, model.parameters())
    )
    target_unchanged = torch.equal(
        target_q_values, torch.tensor([1.01, 0.20])
    ) and not target_q_values.requires_grad

    checks = (
        ("q_values_shape=(2, 2)", q_values_ok),
        ("selected_q_before=[-0.05, 0.6]", selected_ok),
        ("per_item_losses=[1.1236, 0.16]", per_item_ok),
        ("mean_loss=0.6418", loss_ok),
        ("loss_is_scalar=True", isinstance(loss, torch.Tensor) and loss.ndim == 0),
        ("both_predictions_moved_toward_targets=True", moved_toward),
        ("selected_q_after=[0.0613, 0.55]", selected_after_ok),
        ("loss_after=0.5113", loss_after_ok),
        ("parameters_changed=True", parameters_changed),
        ("target_values_unchanged=True", target_unchanged),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")
    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：一批经验已通过平均损失完成一次参数更新")
    elif result is False:
        print()
        print("还有检查未通过，请只修改 batch_loss_update() 中的 TODO")


if __name__ == "__main__":
    main()
