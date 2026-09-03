---
title: ReLU与梯度门控
aliases:
  - ReLU backward
  - ReLU gradient gate
tags:
  - knowledge-graph/core
  - reinforcement-learning/neural-network
  - reinforcement-learning/pytorch
status: learning
created: 2026-09-03
updated: 2026-09-03
related:
  - "[[概念/隐藏层与非线性]]"
  - "[[概念/自动求导与梯度]]"
  - "[[概念/所选动作Q值与索引]]"
  - "[[062-gradient-through-output-and-relu]]"
---

# ReLU 与梯度门控

ReLU 在前向时把负数截成 0，在反向时根据同一次前向的输入正负决定梯度能否通过：

$$
ReLU(z)=\max(0,z)
$$

$$
\frac{\partial ReLU(z)}{\partial z}=
\begin{cases}
1,&z>0\\
0,&z\le 0
\end{cases}
$$

因此，ReLU 后收到的上游梯度还不是 ReLU 前的梯度。它必须逐项乘本轮的 ReLU 门：

```text
ReLU 后梯度 × 正负门 → ReLU 前梯度
```

固定例子：

```text
ReLU 前 z       = [ 0.2, -0.5,  0.2]
ReLU 后收到梯度 = [ 0.8, -0.4, -1.6]
ReLU 门          = [ 1.0,  0.0,  1.0]
ReLU 前梯度      = [ 0.8,  0.0, -1.6]
```

中间位置的 `-0.4` 被截断，不是因为数值为负，而是因为对应的前向输入 `z[1]` 为负，使局部导数为 0。

## 三类零梯度不要混淆

- 未选动作的输出梯度为 0：该输出没有接入当前 loss；
- 关闭的隐藏路径梯度为 0：ReLU 的本轮局部导数为 0；
- 某列权重梯度为 0：本轮对应输入可能恰好为 0。

一次零梯度不代表该参数永远不能学习。下一条观察可能改变输入值，也可能让 ReLU 门重新打开。

## PyTorch 观察接口

参数是叶子张量，反传后默认保留 `.grad`。隐藏中间张量是非叶子张量；反传仍会经过它，但若要在反传后读取其 `.grad`，需要提前调用 `retain_grad()`。

`retain_grad()` 只要求保存经过此处的梯度，不会改变前向值、反向公式或参数。

## 对应课程与代码

- [[062-gradient-through-output-and-relu|梯度怎样穿过输出层和 ReLU 回到隐藏层]]
- `exercises/cartpole_relu_gradient_flow.py`

> [!warning] 当前边界
> 教师已完成固定数字推导，学习者练习尚未完成。当前只检查 `backward()` 写入梯度，不包含 optimizer 参数更新或完整 CartPole 训练。
