---
title: Python函数怎样完成一次最小预测
aliases:
  - 神经网络Python前置
  - 输入计算输出
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/python
status: completed
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[037-cartpole-interface-random-baseline]]"
  - "[[概念/Python函数模型]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[学习主页]]"
---

# Python 函数怎样完成一次最小预测

## 本课目标

长期方向仍是用 DQN 学习 Chromium B.S.U.，之后再沿另一条路线进入 Go2。本课暂停 PyTorch，只理解一件事：Python 函数怎样接收观察和参数，经过计算返回一个预测值。

## 先把模型看成加工机器

先不谈神经网络。可以把一个模型想成加工机器：

```text
输入数字 → 按内部规则计算 → 输出数字
```

Python 函数正好能表达这个过程：

```python
def predict_q(observation, weight, bias):
    weighted_observation = observation * weight
    prediction = weighted_observation + bias
    return prediction
```

逐行看：

1. `def predict_q(...):` 定义一个名为 `predict_q` 的函数；
2. 括号里的 `observation、weight、bias` 是调用函数时交给它的输入；
3. 函数内部先把观察乘以权重；
4. 再加上偏置；
5. `return prediction` 把结果交还给调用者。

调用时写：

```python
result = predict_q(observation=-0.02, weight=3.0, bias=0.1)
```

这句话不是在训练，只是把三个具体数字送进函数，并把返回结果保存到 `result`。

## 权重和偏置为什么存在

先用普通话说：

- 权重决定某项观察对输出影响多大、方向是正还是负；
- 偏置是在观察贡献之外，再给输出加上的一个可调起点。

理解这句话后，再把本例写成计算：

```text
观察贡献 = 观察 × 权重
预测值   = 观察贡献 + 偏置
```

代入本课数字：

```text
观察贡献 = -0.02 × 3.0 = -0.06
预测值   = -0.06 + 0.10 = 0.04
```

## 可见示例

运行：

```bash
.venv/bin/python -m examples.python_function_model
```

输出：

```text
observation=-0.02
weight=+3.00
weighted_observation=-0.06
bias=+0.10
prediction=+0.04
```

这就是神经网络最底层思想的缩小版：使用参数把输入组合成输出。真正的神经网络只是同时处理更多输入、保存更多参数，并把多个这样的计算连接起来。

## 本课编程练习

打开并只完成其中的 `TODO`：

- `exercises/python_function_q_prediction.py`

运行：

```bash
.venv/bin/python -m exercises.python_function_q_prediction
```

示例只处理一项观察；练习要求你把“杆角度”和“杆角速度”两项观察分别乘以对应权重，再加上偏置。这是在旧知识上多组合一项输入，不是照抄示例。

## 学习者练习结果

学习者完成了 `predict_right_q()`：从 `observation` 和 `weights` 中分别读取两项，将每项观察乘以对应权重，再加上 `bias` 并返回。

实际运行三组输入全部通过：

```text
场景1 prediction=+0.500 expected=+0.500 PASS
场景2 prediction=+1.600 expected=+1.600 PASS
场景3 prediction=+0.500 expected=+0.500 PASS
练习通过：你已经用一个 Python 函数组合两项观察并返回预测值
```

代码不是固定返回某个测试答案，而是根据传入的观察和权重计算，因此本课达到完成条件。

## 当前证据边界

> [!success] 启动检查
> 标准库示例和学习者练习均已实际运行；三组练习场景全部通过，测试确认函数会使用不同输入得到不同预测，并且调用函数本身不会修改传入的观察和参数。

> [!warning] 尚未验证
> 尚未学习 Python 类、PyTorch、张量、多层神经网络或 DQN 训练。

## 一句话总结

最小模型可以先写成一个普通 Python 函数：它接收观察和参数，按规则计算，再用 `return` 返回预测。

## 关联

- 前置：[[037-cartpole-interface-random-baseline|CartPole 环境接口与随机策略基线]]
- 概念：[[概念/Python函数模型]]
- 后续关系：[[概念/神经网络参数与预测]]
- 学习入口：[[学习主页]]
