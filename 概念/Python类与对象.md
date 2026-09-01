---
title: Python类与对象
aliases:
  - class object self
  - Python模型对象
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/python
status: learned
created: 2026-09-01
updated: 2026-09-01
related:
  - "[[概念/Python函数模型]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[概念/PyTorch张量]]"
  - "[[039-python-class-stores-parameters]]"
---

# Python 类与对象

类描述一种对象要保存的数据和能够执行的方法；对象是根据类创建的具体实例；`self` 表示方法当前正在操作的那个对象。

模型使用类以后，可以在 `__init__()` 中把权重和偏置保存到 `self`，再在 `predict()` 中重复读取：

```text
创建对象 → __init__ 保存参数 → predict 接收新观察 → 使用自身参数预测
```

## 职责边界

- 类是结构定义，对象才是实际保存参数的实例；
- 两个对象可以由同一个类创建，但各自保存不同参数；
- `self.weight` 是对象内部数据，不需要每次调用 `predict()` 时重新传入；
- 把参数放进对象仍不等于训练，参数更新属于后续课程。

## 对应课程与代码

- [[039-python-class-stores-parameters|Python 类怎样保存模型参数]]
- `examples/python_class_model.py`
- `exercises/python_class_q_model.py`

> [!success] 学习者证据
> 学习者完成 `__init__()` 参数保存和 `predict()` 预测，三组场景及两个对象参数独立性检查全部通过。
