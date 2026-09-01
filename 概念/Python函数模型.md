---
title: Python函数模型
aliases:
  - 输入计算输出
  - 函数式预测
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/python
status: learning
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[概念/观察与动作]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[038-python-function-as-model]]"
---

# Python 函数模型

Python 函数可以表达最小的预测过程：接收观察和参数，在函数内部计算，然后用 `return` 把预测值交还给调用者。

```text
函数参数接收数字 → 局部变量保存中间结果 → return 返回输出
```

这里的“函数参数”是 Python 调用时传入的名字；其中 `weight` 和 `bias` 又承担模型可调参数的角色。两种“参数”语境相关，但并不完全相同。

## 职责边界

- 调用预测函数只计算结果，不等于训练；
- `return` 返回的是一个新结果，不会自动修改输入；
- 权重和偏置怎样更新属于后续学习，本节点只解释怎样使用它们；
- PyTorch 网络会把大量类似计算组织起来，但不改变“输入经过参数计算得到输出”的基本关系。

## 对应课程与代码

- [[038-python-function-as-model|Python 函数怎样完成一次最小预测]]
- `examples/python_function_model.py`
- `exercises/python_function_q_prediction.py`

