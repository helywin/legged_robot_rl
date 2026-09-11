---
title: 经验池与CPU更新吞吐优化
status: learning
tags: [强化学习, Chromium, 性能]
---

## 问题与假设

约2000更新/秒时，增加进程不能保证继续加速。检查Python数据写入与小网络频繁派发的开销。
性能实验不改奖励、每条可学习经验对应一次更新的比例、批量32和环境调度顺序。
学习者已将task.py中的击毁奖励改为×10，本轮保留，所有性能组使用相同修改；
它尚未独立版本化，不能将本次权重当作旧v5训练可复现结果。

## 实现

1. TensorReplay保留CPU张量存储，通过共享内存的NumPy视图写入，
   避免每条经验建立两个临时张量及多次标量张量赋值。
   保留相同Random.sample顺序、FIFO映射、float32/int64/bool类型和独立抽样批次。
   pyproject.toml直接声明NumPy；环境已安装2.5.2，本轮没有安装或升级环境。
2. 新增可选ScriptedSGDAgent：固定前向、目标、MSE、自动求导及普通SGD更新编译执行。
   使用autograd.grad返回梯度，编译循环执行p.add_(grad, alpha=-lr)，不写.grad。
   仅支持CPU float32、一个参数组、无动量/权重衰减的普通SGD；不支持配置明确拒绝。
   学习版Agent保留；目标网络仍在外部按相同次数同步，网络共享同一参数，权重格式不变。

成功条件：固定配置重复计时更快，并且真实训练的每步日志和所有最终参数完全一致。
吞吐改善不代表更少经验收敛，也不证明躲避行为改善。

## 环境与配置

Intel Core Ultra 9 285HX；Python3.14；torch2.13.0+cpu；CPU单线程更新；
32个无窗口原生进程。CUDA编译支持为None、is_available=False。
nvidia-smi可见RTX5070Ti Laptop 12227MiB，但当前虚拟环境不是CUDA版PyTorch，
因此没有进行CUDA实测，没有改驱动或安装GPU依赖。

编译后端在本机数值与运行对照通过，但PyTorch明确警告torch.jit.script不支持Python3.14+，
可能不兼容后续变化。因此作为显式--update-backend scripted实验入口，
默认保持eager。不将本机通过描述为官方支持。

```json
{
  "max_updates": 10000,
  "update_backend": "eager",
  "num_envs": 32,
  "task_version": "task-v5-shield-damage",
  "episode_limit": 1000,
  "capacity": 10000,
  "batch_size": 32,
  "learning_starts": 256,
  "target_sync_every": 100,
  "epsilon": 0.2,
  "gamma": 0.99,
  "learning_rate": 0.001,
  "game_seed": 31,
  "network_seed": 7,
  "replay_seed": 7,
  "exploration_seed": 11
}
```

## 完整命令

```bash
.venv/bin/python -m cProfile -o /tmp/chromium-training.prof exercises/chromium_dqn/train.py --run --max-updates 4000 --num-envs 32
.venv/bin/python experiments/2026-09-11-chromium-replay-throughput/benchmark.py
.venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -p 'test_fast_update.py' -v
.venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -p 'test_parallel_training.py' -v
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 500 --num-envs 8 --update-backend scripted
```

benchmark.py交替先后顺序、顺序执行6次训练，避免组间争抢CPU，
每组10000更新、32环境，生成模型后核对全部steps.jsonl字节及online每个参数。
每次从头初始化；网络seed7，replayseed7，探索seed11，游戏初始seed31，
环境i第k局seed31+i+k×32，每局1000决策。

## 结果与产物

经验池单独改动：3次10000更新，中位数5.281435→5.180799秒，吞吐约+1.9%；
首组最终所有网络参数完全一致，优化幅度很小。

保留新版经验池，再比较两种更新后端：

|重复|普通更新秒|编译更新秒|日志/权重|
|---|---|---|---|
|1|5.134626|4.541762|完全一致|
|2|5.210451|4.401471|完全一致|
|3|5.181062|4.337587|完全一致|

中位数5.181062→4.401471秒，即1930.11→2271.97更新/秒，吞吐约+17.7%。
耗时包括原生进程启动、训练、日志和关闭，不含导入/建网络；
编译后端初始化另记backend_init_seconds，本轮为约0.009～0.022秒。
训练首次调用编译内核的预热包含在耗时内；这些是短预算吞吐，不保证长训练同速。

环境数附加扫描（各一次，不与同配置对照混算）：

|环境数|编译更新/秒|
|---|---|
|1|2027.7|
|4|2104.3|
|8|2203.1|
|16|2323.6|

这说明环境越多未必越快。16环境单次2323.6/秒，32环境三次中位数2272.0/秒；
不足以认定16一定最优，也没有把改变环境数的权重当作相同训练轨迹。

原始数据在exercises/chromium_dqn/runs/：

- replay-speed-before.json / replay-speed-after.json：经验池单独对照及训练路径。
- compiled-speed-comparison.json：六次结果、训练路径、源码哈希与精确一致断言结果。
- compiled-env-sweep.json：环境数量扫描。
- 各训练目录保留config.json、steps.jsonl、summary.json、policy.pt，生成产物不提交。

六组训练目录：

- eager repeat0: `train-13488e7f1d1642618155d749079f8e37`
- scripted repeat0: `train-3dd29104a6f4449cb5a172d64d11f924`
- scripted repeat1: `train-152c27077ed948b99f37150b3ba621e5`
- eager repeat1: `train-7489290d600d407baaf75abeaa9ba036`
- eager repeat2: `train-635998c12c4f478da660cbc5d6c187b6`
- scripted repeat2: `train-ede7b157d45548c484fbe98835341e01`

## 验证边界与下一步

200步自动测试逐次比较终止/截断混合批次、目标同步和参数；另有1000步手工固定数据测试通过。
3组真实训练对照每组10000更新，日志与最终权重精确一致。
经验池环回、采样顺序、批次不共享存储、并行精确预算/独立reset测试通过。
保留用户未提交的task.py奖励修改；未把按旧奖励预期编写的检查改成新的固定答案。
没有新增C++改动，没有GUI或策略效果验收。

当前收益约18%，没有获得数量级提升。下一步若研究CUDA，需要独立准备兼容GPU的PyTorch环境，
再以相同小批次做CPU/GPU实测；现有证据不能承诺GPU一定更快。
