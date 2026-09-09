---
title: Python动作与游戏逻辑时间
status: learning
tags:
  - chromium
  - experiment
---

# Python动作与游戏逻辑时间

## 问题与原理

上一阶段能读取状态，但两次读取之间游戏一直在跑。假如Python计算慢了，飞机和敌人就会多走一些，无法说清“这个动作到底执行了多久”。

现在改为：**Python发出动作，游戏完整执行指定步数，再返回结果；没有下一条指令就等待。** step在这里是一次有明确边界的推进，不是截图或按键模拟。

例如请求向右1步：控制累积量从0开始，首次按下加5；保持输入再加`2+5×0.4=4`；衰减后为`9×0.7=6.3`。原游戏把它转成整数6，再乘默认移动比例0.03，所以x从0变成0.18。下一步继续右移不会再加首次按下的5，算得7.574，取整数7，所以x变为0.39。

这解释了两个不同的“停”：不调用step时，整个游戏冻结；调用`step(Action.IDLE)`时，游戏仍推进，只是释放按键，移动累积量继续衰减。例如上面的7.574变成5.3018，飞机还会再移动0.15。不能把IDLE理解成暂停世界。

```python
from chromium_rl import Action, GameClient

with GameClient(synchronous=True) as game:
    result = game.step(Action.RIGHT_FIRE, ticks=1)
    print(result.snapshot.player.position)
```

GameClient拥有一个独立游戏进程；synchronous=True让它直接进入仅第一关的同步控制模式。RIGHT_FIRE同时声明方向和开火状态。ticks=1要求完整执行一步；result.snapshot是这一步结束后的状态。不是神经网络在控制，这里尚无训练。

每步按原游戏50fps参考规则执行，speedAdj=1，标称0.02秒。它不是任意放大的物理dt。Python等待0.1秒后才发下一条指令，游戏不会因此多跑5步。

## 假设与唯一变化量

保持动作序列与步数不变，只改变Python两条指令之间的等待时间。预期画面播放变慢，但早期每个tick的位置不变；固定2秒不发指令时，快照不变。

## 环境与配置

- 1个原生游戏GUI；不是Isaac Lab。训练迭代数0，无网络或检查点。
- 固定内部步数1；初始第一关，不执行菜单帧。
- 随机种子尚未支持，所以不比较整局分数，不声称完整确定性。
- 基线等待0.02秒；对照等待0.15秒；其他参数不变。
- 工程状态：C阶段部分实现。每步仍执行原版绘图；不能称为无渲染训练。

## 命令与亲手对照

先看一次演示，不需要输入法或按游戏键：

```bash
.venv/bin/python third_party/chromium-bsu-rl/examples/watch_steps.py
```

想亲手验证时间关系时，分别运行下面两次，只改--delay：

```bash
.venv/bin/python third_party/chromium-bsu-rl/examples/watch_steps.py --delay 0.02
.venv/bin/python third_party/chromium-bsu-rl/examples/watch_steps.py --delay 0.15
```

不要求填接口或通过固定答案检查器。观察两个真实现象：慢放是否增加了tick；输出“No requests”时，整个游戏是否停止变化。成功条件是等待只改变观看速度、不增加游戏步数；如画面仍自动推进，保留输出并先排查启动模式。实验所需说明也在脚本开头。

## 产物、观察与结果

- 终端输出动作、episode_tick、参考模拟时间、xy及停顿前后快照比较。
- 教师本地构建日志：third_party/chromium-bsu-rl/build/stage-c-build.log。
- 本地截图：/tmp/chromium-rl-stage-c-game.png，已看到右移后的飞机与子弹，不提交截图。
- 教师原生接口检查：初始等待不推进、连续右移0.18→0.39、释放后0.54、斜向移动、精确多步、错误不推进和关闭清理通过。
- 一次未固定seed的空动作诊断在tick852结束；最后50步请求只执行2步，之后拒绝step且状态不变。该结果不是训练成绩或可复现基准。
- 学习者本节对照结果：待报告；状态保持learning。运行教师演示不记作已完成训练。

## 结论与下一步

已验证同步GUI接口的动作/步数边界，不代表加速训练、确定性或完整环境完成。当前渲染会改变无敌计数、部分敌人行为和随机游标，因此不能直接删掉绘图调用。下一工程步骤是拆开这些副作用，再验证seed/reset与无渲染运行；还不连接DQN。

关联：[[概念/游戏内部状态与策略接口]]；上一项：[[experiments/2026-09-07-chromium-live-snapshot/README]]。
