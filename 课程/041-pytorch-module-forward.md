---
title: nn.Module怎样组织参数和前向计算
aliases:
  - PyTorch Module 入门
  - forward 前向计算
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[040-pytorch-tensor-prediction]]"
  - "[[概念/PyTorch模块与参数]]"
  - "[[概念/PyTorch张量]]"
  - "[[概念/Python类与对象]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[学习主页]]"
---

# nn.Module 怎样组织参数和前向计算

## 本课目标

长期方向仍是用 DQN 学习 Chromium B.S.U.，之后再沿另一条路线进入 Go2。本课只把已经学过的 Python 类和张量计算组织成 PyTorch 标准模块；不学习多层网络、损失、反向传播或参数更新。

## nn.Module 仍然是 Python 类

普通模型类：

```python
class OneInputQModel:
    ...
```

PyTorch 模型类：

```python
class OneInputQModule(nn.Module):
    ...
```

括号里的 `nn.Module` 表示新类在普通 Python 类能力之外，继承 PyTorch 提供的模型管理能力，例如登记参数、遍历参数和保存模型状态。

## 三个新职责

1. `nn.Module`：PyTorch 模型的基础类；
2. `nn.Parameter`：告诉 PyTorch“这个张量属于模型参数”；
3. `forward()`：描述观察怎样使用参数得到输出。

最小示例：

```python
class OneInputQModule(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, observation):
        return observation * self.weight + self.bias
```

`super().__init__()` 先初始化父类 `nn.Module`，之后赋给 `self` 的 `nn.Parameter` 才能被正确登记。

## 为什么调用 model 而不是 forward

创建模型后通常写：

```python
prediction = model(observation)
```

PyTorch 会自动转到模型的 `forward(observation)`。不直接调用 `forward()`，是因为 `model(...)` 还保留了 PyTorch 在前向计算前后管理其他功能的入口。

## VS Code 为什么可能把结果显示成 Any

`forward()` 的返回类型虽然写成了 `torch.Tensor`，但 `model(observation)` 会先经过 PyTorch 基类的 `nn.Module.__call__`。Pylance/Pyright 看到这个通用入口时，可能把返回值推断为 `Any`，导致后面的 `loss` 也失去类型提示。

教学模型因此显式补充一个带类型的调用入口：

```python
def __call__(self, observation: torch.Tensor) -> torch.Tensor:
    return super().__call__(observation)
```

这里必须继续调用 `super().__call__()`：它只向 IDE 补充输入输出类型，同时保留 PyTorch 的前向钩子和完整模块调用流程，最后仍会进入 `forward()`。不能把它改成直接调用 `self.forward()`。

运行：

```bash
.venv/bin/python -m examples.pytorch_module_prediction
```

实际输出：

```text
is_nn_module=True
registered_parameter=weight value=+3.00
registered_parameter=bias value=+0.10
observation=-0.02
prediction=+0.04
prediction_type=Tensor
```

`named_parameters()` 能找到 `weight` 和 `bias`，说明它们不只是普通属性，而是已经登记的模型参数。

## 本课训练

打开并只完成两个 `TODO`：

- `exercises/pytorch_module_q_prediction.py`

运行：

```bash
.venv/bin/python -m exercises.pytorch_module_q_prediction
```

练习要求把两项权重与偏置登记进模块，并在 `forward()` 中完成前一课相同的预测。程序会检查参数登记、三组预测和前向计算不修改参数。

## 学习者练习与纠错结果

学习者先正确完成了 `nn.Module` 初始化和两项 `nn.Parameter` 登记，但第一次在 `forward()` 中使用逐项乘法，得到长度为 2 的输出：

```text
tensor([0.7000, 0.3000])
```

这暴露出“两个观察贡献尚未合并成一个 Q 值”。学习者随后改用张量点积，把两项贡献相加为标量。

最终实际运行结果：

```text
模型是 nn.Module：PASS
参数已登记：PASS
场景1 prediction=+0.500 expected=+0.500 PASS
场景2 prediction=+1.600 expected=+1.600 PASS
场景3 prediction=+0.500 expected=+0.500 PASS
前向预测没有修改参数：PASS
练习通过：nn.Module 已登记参数并通过 forward 完成预测
```

参数登记、标量输出和参数不变性全部通过，因此本课达到完成条件。

## 当前证据边界

> [!success] 启动检查
> 教师示例、学习者练习和自动测试确认 `nn.Module` 能登记参数，且调用模型对象会执行标量前向预测。

> [!warning] 尚未验证
> 还没有损失、自动求导、优化器、参数更新或 DQN。

## 一句话总结

`nn.Module` 用 Python 类组织模型，`nn.Parameter` 登记参数，`forward()` 规定输入怎样变成输出。

## 关联

- 前置：[[040-pytorch-tensor-prediction|PyTorch 张量预测]]
- 概念：[[概念/PyTorch模块与参数]]
- 张量：[[概念/PyTorch张量]]
- Python 类：[[概念/Python类与对象]]
- 后续关系：[[概念/神经网络参数与预测]]
- 学习入口：[[学习主页]]
