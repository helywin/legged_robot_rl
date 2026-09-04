---
title: 第一次真实CartPole DQN冒烟训练
aliases:
  - first real CartPole DQN training
  - CartPole DQN smoke training
tags:
  - reinforcement-learning/dqn
  - reinforcement-learning/cartpole
  - reinforcement-learning/experiment
status: learning
created: 2026-09-04
updated: 2026-09-04
related:
  - "[[065-three-clocks-in-dqn-training]]"
  - "[[概念/DQN小型基准]]"
  - "[[概念/DQN训练的三种计数]]"
  - "[[概念/训练与评估]]"
  - "[[experiments/2026-09-04-cartpole-dqn-smoke/README]]"
  - "[[概念/DQN完整训练流程与公式]]"
  - "[[学习主页]]"
---

# 第一次真实 CartPole DQN 冒烟训练

## 本课终于训练什么

前面的固定数字只验证局部机理。这一课第一次把已经学过的部件接到真实 `CartPole-v1`：

```text
真实环境产生观察和奖励
→ epsilon-greedy 选择动作
→ 经验进入 replay buffer
→ 随机抽取一批经验
→ 在线网络预测实际动作 Q 值
→ 目标网络构造 target
→ loss → backward → optimizer.step
→ 定期同步目标网络
→ 关闭探索和更新，独立评估
```

它不是固定答案检查器。环境会真的推进，经验会真的积累，网络会连续更新 29,001 次，最终用坚持步数判断有没有学到控制能力。

## 1. 实验问题和唯一比较

本课只比较同一个网络的两个时刻：

| 时刻 | 网络参数 | 探索 | 参数更新 | 用途 |
| --- | --- | --- | --- | --- |
| 训练前冻结评估 | 随机初始化 | 关闭 | 关闭 | 测量起点 |
| 30,000 环境步训练 | 持续变化 | 开启并逐渐减少 | 开启 | 产生学习 |
| 训练后冻结评估 | 训练后参数 | 关闭 | 关闭 | 测量结果 |

训练前后评估使用相同的 20 个独立种子，因此改变的是网络是否经历训练，而不是评估初始状态。

历史随机策略基线 `21.40` 是另一个参照：它每步随机选动作；训练前冻结网络则始终选择自己当前 Q 值最大的动作。两者都没有学习，但不是同一种策略。

## 2. 为什么叫冒烟训练

“冒烟训练”表示先用有限预算确认完整系统确实能学，而不是立即证明已经达到最终目标。

本课预先规定：

```text
20 个冻结评估回合平均回报 >= 200：冒烟通过
```

最终验收仍保持第 036 课的更严格标准：

```text
100 个独立评估回合平均回报 >= 475
其中至少 90 回合达到完整 500 步
```

因此得到 `200～474` 说明训练闭环有效，但不能说 CartPole 已经最终通过。

## 3. 参数按职责分组，不是一堆独立旋钮

你运行时不需要填写任何参数。脚本把它们集中在 `TrainingConfig`，这里按职责解释。

### 数据采集

```text
total_environment_steps = 30000
epsilon：1.0 → 0.05，前 20000 步线性下降
```

环境一共执行 30,000 次动作。早期大量随机动作负责探索不同状态；后期仍保留 `0.05` 的小概率探索，避免训练数据只剩当前策略熟悉的动作。

### 经验记忆

```text
replay_capacity = 30000
warmup_steps = 1000
batch_size = 64
```

前 1,000 步先收集经验。达到预填充后，每个环境步从历史经验中随机抽 64 条，共同完成一次更新。

### 网络更新

```text
discount_factor = 0.99
learning_rate = 0.001
optimizer = Adam
```

折扣因子让未终止经验保留大部分未来价值。学习率控制每次参数变化尺度。

Adam 仍然是 optimizer：创建时保存在线参数引用，`backward()` 先把梯度写进 `.grad`，`Adam.step()` 再读取梯度修改参数。它与前面 SGD 的内部步长算法不同，会根据历史梯度调整各参数的实际步幅；但 `loss → grad → step → 新参数` 的接力关系没有改变。本课固定使用它，不把 optimizer 对比混入第一次真实训练。

### 目标稳定与评估

```text
target_sync_interval = 500 次 update
evaluation_episodes = 20
smoke_success_mean_return = 200
```

目标网络不是每步同步，而是在线参数每更新 500 次才复制一次。评估使用单独环境，完全关闭 epsilon 探索和 optimizer 更新。

## 4. 训练循环中真正改变了什么

```plantuml
@startuml
top to bottom direction

start
:当前 observation 进入在线网络;
:epsilon-greedy 选择并执行 action;
:env.step() 产生 reward 和 next observation\nenvironment_step 加 1;
:保存一条 transition;

if (buffer 达到 1000 条?) then (是)
  :随机抽取 64 条经验;
  :在线网络产生 selected Q;
  :目标网络产生无梯度 targets;
  :mean loss → backward;
  :Adam.step() 修改在线参数\nupdate_count 加 1;
  if (update_count 到达 500 的倍数?) then (是)
    :在线参数复制给目标网络;
  else (否)
    :目标网络保持不变;
  endif
else (否)
  :继续积累真实经验;
endif

if (当前 episode 结束?) then (是)
  :记录本局 return;
  :env.reset() 开始下一局;
else (否)
  :继续使用 next observation;
endif

stop
@enduml
```

每一轮中：

- 环境状态由 `env.step()` 改变；
- replay buffer 由 `add()` 改变；
- `.grad` 由 `backward()` 写入；
- 在线参数只由 `Adam.step()` 改变；
- 目标参数只在同步时由复制改变；
- episode 结束时环境重置，但网络和 replay buffer 不重置。

## 5. 怎样读一行训练日志

教师参考运行在第 5,000 个环境步输出：

```text
environment_step=5000
update_count=4001
episodes=212
epsilon=0.762
recent_return=35.00
recent_loss=0.3185
```

逐项含义：

- 环境已经执行 5,000 次动作；
- 第 1～999 步只预填充，第 1,000～5,000 步均更新，所以共有 `5000-1000+1=4001` 次更新；
- 已经结束 212 个 episode；
- 当前仍有约 76.2% 概率随机探索；
- 最近 20 局平均坚持 35 步；
- 最近 100 次随机批量更新的平均 loss 为 0.3185。

这里的 `recent_return` 属于训练期间，动作仍带探索，不能代替冻结评估。

## 6. 为什么训练 loss 上升，策略却明显变好

教师参考运行中：

```text
recent_loss：0.3185 → 4.7614
recent_return：35.00 → 297.00
```

这不是同一个指标发生矛盾：

- `loss` 比较当前批量的 selected Q 和当前 target；
- `return` 统计小车实际坚持了多少环境步；
- 策略活得更久后，未来累计奖励变大，Q 值和 target 的数值尺度也会变大；
- 目标网络每 500 次更新跳动一次，target 不是训练全过程固定的尺子；
- 每次随机抽到的 64 条经验难度不同。

因此不能要求 DQN 的 batch loss 像固定监督学习数据那样一路单调下降。真正的控制能力仍要看关闭探索和更新后的独立环境表现。

## 7. 教师参考运行结果

训练进度：

| environment step | update count | 最近20局平均回报 | epsilon |
| ---: | ---: | ---: | ---: |
| 5,000 | 4,001 | 35.00 | 0.762 |
| 10,000 | 9,001 | 68.75 | 0.525 |
| 15,000 | 14,001 | 232.80 | 0.288 |
| 20,000 | 19,001 | 294.80 | 0.050 |
| 25,000 | 24,001 | 298.25 | 0.050 |
| 30,000 | 29,001 | 297.00 | 0.050 |

相同 20 个独立种子的冻结评估：

```text
训练前平均回报：  9.20
随机策略历史基线：21.40
训练后平均回报：253.60
冒烟通过线：    200.00 PASS
```

这证明本次真实训练闭环产生了明显学习效果。但 `253.60 < 475`，所以它只是冒烟通过，不是最终 CartPole 验收通过。

## 8. 本课训练：亲手接完整闭环

教师参考实现保留在 `train.py`，用于最后排查和对照。你这次先不要照抄它，而是在下面的学习者脚手架中亲手完成核心：

```text
experiments/2026-09-04-cartpole-dqn-smoke/learner_train.py
```

文件内已经写明全部接口、形状、允许使用的 PyTorch 操作和运行方式。只修改 `TODO 1`～`TODO 5`：

| TODO | 阶段 | 你要建立的因果关系 |
| ---: | --- | --- |
| 1 | 网络 | `[batch,4]` 观察怎样变成 `[batch,2]` Q 值 |
| 2 | 探索计划 | 环境步怎样决定当前 epsilon |
| 3 | 数据采集 | epsilon 怎样在随机动作和最大 Q 动作间选择 |
| 4 | 参数学习 | 一批经验怎样成为 loss、grad 和新在线参数 |
| 5 | 主时间线 | 环境交互、保存经验、抽样更新、目标同步和 reset 怎样连接 |

先打开总流程和关键知识文档，对照着写：[[概念/DQN完整训练流程与公式]]。

完成后从仓库根目录运行：

```bash
.venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/learner_train.py
```

当前预算是单环境 30,000 步，教师参考耗时约 10 秒。你的实现成功后应生成：

```text
artifacts/cartpole-dqn-from-scratch/online-network.pt
artifacts/cartpole-dqn-from-scratch/online-network.onnx
artifacts/cartpole-dqn-from-scratch/metrics.json
```

其中 ONNX 只保存训练后在线网络的前向推理：

```text
observation: float32[batch_size, 4]
→ q_values: float32[batch_size, 2]
```

它不包含环境、epsilon、replay buffer、target 网络、loss、`backward()` 或 optimizer。外部工具得到两个 Q 值后，仍由调用方取最大值对应的动作。

### 第一个成功条件：真实冻结评估

成功不是每个浮点数与教师完全相同，而是最终显示：

```text
冒烟通过线：200.00 PASS
```

### 第二个成功条件：ONNX 转换没有改变输出

```bash
.venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/verify_onnx.py
```

脚本会把同一批三条观察分别交给 PyTorch 和 ONNX Runtime。成功时看到：

```text
ONNX 结构检查：PASS
数值对照：PASS
```

### 最后用 UI 看策略实际控制

```bash
.venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/watch.py
```

这个窗口加载你的 `.pt` 检查点，运行三个固定种子的贪心回合：无 epsilon 探索、无 replay buffer、无 `backward()`、无参数更新。窗口负责让控制行为可见；冻结评估数字负责判断是否通过，两者不能互相替代。

运行后把训练末尾的冻结评估、ONNX 数值对照，以及 GUI 中是否正常保持平衡发给我。本课在看到你的实际运行证据前保持 `learning`。

## 当前边界

> [!warning]
> 教师参考只验证单训练种子、30,000 环境步和 20 个冻结评估种子。学习者实现、学习者 ONNX 产物和本机原生 GUI 尚待验证；也尚未达到最终 100 回合验收线，没有多种子复现、Isaac Lab 或真机证据。

## 一句话总结

真实 DQN 训练把环境采集、回放抽样、在线参数更新、目标同步和 episode 重置放进同一时间线；是否学会不能只看 loss，要用关闭探索和更新后的独立回报与训练前基线比较。

## 关联

- 三条时间线：[[065-three-clocks-in-dqn-training|DQN 训练为什么有三种计数]]
- 实验记录：[[experiments/2026-09-04-cartpole-dqn-smoke/README|第一次真实 CartPole DQN 冒烟训练]]
- 基准：[[概念/DQN小型基准]]
- 评估：[[概念/训练与评估]]
