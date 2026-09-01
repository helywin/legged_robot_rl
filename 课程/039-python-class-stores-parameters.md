---
title: Python类怎样保存模型参数
aliases:
  - Python 模型对象
  - class object self
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/python
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[038-python-function-as-model]]"
  - "[[概念/Python类与对象]]"
  - "[[概念/Python函数模型]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[学习主页]]"
---

# Python 类怎样保存模型参数

## 本课目标

长期方向仍是用 DQN 学习 Chromium B.S.U.，之后再沿另一条路线进入 Go2。本课只学习怎样用 Python 类创建一个模型对象，让权重和偏置保存在对象内部；不使用 PyTorch，也不更新参数。

## 为什么函数还不够方便

上一课每次预测都要传入所有内容：

```python
predict_q(observation, weight, bias)
```

但模型的观察会不断变化，参数通常会连续使用很多次。更自然的结构是：

```text
创建模型时保存参数
        ↓
每次预测只交给它新的观察
```

## 三个 Python 词

- **类（class）**：描述一种对象应该保存什么、能做什么；
- **对象（object）**：按照类创建出来的一个具体模型；
- **self**：在类的方法内部，表示“当前这个对象自己”。

最小代码：

```python
class OneInputQModel:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def predict(self, observation):
        return observation * self.weight + self.bias
```

`__init__()` 在创建对象时运行：

```python
model = OneInputQModel(weight=3.0, bias=0.1)
```

两次赋值把参数放进 `model` 对象：

```text
model.weight = 3.0
model.bias   = 0.1
```

之后预测只需要传入新观察：

```python
first = model.predict(-0.02)
second = model.predict(0.05)
```

两个调用都读取同一个 `model` 内保存的参数。

## 可见示例

运行：

```bash
.venv/bin/python -m examples.python_class_model
```

预期输出：

```text
stored_weight=+3.00
stored_bias=+0.10
observation=-0.02 prediction=+0.04
observation=+0.05 prediction=+0.25
```

## 本课训练

打开并只完成两个 `TODO`：

- `exercises/python_class_q_model.py`

运行：

```bash
.venv/bin/python -m exercises.python_class_q_model
```

练习要求把上一课的两项观察预测放进类中，并额外验证两个模型对象可以保存不同参数、互不覆盖。

## 学习者练习结果

学习者在 `__init__()` 中把 `weights` 和 `bias` 保存为 `self.weights` 与 `self.bias`，并在 `predict()` 中读取对象自身参数完成预测。

实际运行结果：

```text
参数保存在对象内部：PASS
场景1 prediction=+0.500 expected=+0.500 PASS
场景2 prediction=+1.600 expected=+1.600 PASS
场景3 prediction=+0.500 expected=+0.500 PASS
两个对象参数互不覆盖：PASS
练习通过：模型对象能够保存并使用自己的参数
```

这同时证明同一个类创建的两个对象可以保存各自的参数，因此本课达到完成条件。

## 当前证据边界

> [!success] 启动检查
> 教师示例、学习者练习和自动测试确认普通 Python 对象可以保存参数并重复预测，且两个对象参数互不覆盖。

> [!warning] 尚未验证
> 本课不是 PyTorch、神经网络训练、反向传播或 DQN。

## 一句话总结

Python 类让模型对象把参数保存在 `self` 中，之后每次预测只需要接收新的观察。

## 关联

- 前置：[[038-python-function-as-model|Python 函数最小预测]]
- 概念：[[概念/Python类与对象]]
- 函数基础：[[概念/Python函数模型]]
- 后续关系：[[概念/神经网络参数与预测]]
- 学习入口：[[学习主页]]
