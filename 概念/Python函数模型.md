---
title: Python函数模型
aliases:
  - 输入计算输出
  - 函数式预测
tags:
  - reinforcement-learning/neural-network
  - reinforcement-learning/python
status: learned
created: 2026-09-01
updated: 2026-09-12
related:
  - "[[概念/观察与动作]]"
  - "[[概念/神经网络参数与预测]]"
  - "[[概念/Python类与对象]]"
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
- `return` 把对象交还调用者，本身不保证创建新对象，也不保证函数没有修改输入；本页早期预测练习约定只读输入；
- 权重和偏置怎样更新属于后续学习，本节点只解释怎样使用它们；
- PyTorch 网络会把大量类似计算组织起来，但不改变“输入经过参数计算得到输出”的基本关系。

## 返回值与修改输入，是两件事

下面是独立的标准库机制示意，可以先预测结果：

```python
def append_and_return(values):
    values.append(3)
    return values

original = [1, 2]
result = append_and_return(original)
print(original)          # 预期：[1, 2, 3]
print(result is original)  # 预期：True
```

`values` 指向调用者传入的列表；`append` 修改同一个列表；`return values` 返回的仍是它；`is` 检查是否为同一个对象。因而不能把“函数有返回值”当作输入不变的证明。模型预测函数不改输入，是实现和接口契约，需要实际检查。

## 对应课程与代码

- [[038-python-function-as-model|Python 函数怎样完成一次最小预测]]
- `examples/python_function_model.py`
- `exercises/python_function_q_prediction.py`

> [!success] 学习者证据
> 学习者完成两项观察、两项权重和偏置的预测函数，三组可见场景全部通过。
