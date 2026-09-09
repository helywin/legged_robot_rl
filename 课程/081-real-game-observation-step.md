---
title: 真实游戏执行动作后观察怎样更新
status: completed
created: 2026-09-09
tags:
  - reinforcement-learning/experiment
related:
  - "[[080-observation-information-boundary]]"
  - "[[078-combined-game-observation]]"
---

# 真实游戏执行动作后观察怎样更新

当前先收敛到接口小目标：真实游戏动作能否在学习者写好的17项观察中反映。它是训练任务之前的数据验收，不是最终生存或通关任务。不扩大字段，保持K=2和已有尺度。

## 原理与手算预期

此前传入的是教学样本；现在传入真实StepResult.snapshot。两者提供编码所需的同名字段，所以原编码逻辑可直接读取，不需要将真实状态重新抄成另一份样本。已有类型注解主要服务阅读/静态检查，不会把真实对象自动转换成教学类。

调用链：真实game.step(action)执行一步 → result.snapshot保存该步结束后的状态 → build_with_resources读取它 → 17项观察。第0项应等于当前世界x/10；编码只读取，不再次推进游戏。

例如下一步原生x=0.18，则观察第0项应为0.18/10=0.018；执行5步后若x=1.26，则该项为0.126。实际浮点值可能显示为0.17999999等小偏差，这是数值表示，不表示额外移动。

## 唯一变化量与亲手实验

入口：[[experiments/2026-09-09-chromium-observation-step/README]]。每次1个GUI、共同初始化1个IDLE tick，再执行5个固定单tick动作；0次网络更新，预计十几秒内结束。唯一人为改变项为IDLE与RIGHT，不开火，不改等待或步数。各进程没有固定seed，不能把整局随机状态当严格一致。

```bash
.venv/bin/python experiments/2026-09-09-chromium-observation-step/run.py --action idle
.venv/bin/python experiments/2026-09-09-chromium-observation-step/run.py --action right
```

脚本已完成，学习者实际运行两组，比较世界x与观察x的变化以及窗口运动。有效实作是执行真实动作对照并报告现象，不是照抄教师结果。

show读取同一snapshot生成观察，检查17项、有限float和x/10，再打印世界位置、飞机四项、敌方子弹数量与资源。每次game.step返回后调用show，因此不会把旧观察与新状态拼接；time.sleep仅留出观看时间。with退出关闭本脚本拥有的原生进程。

## 本轮教师实际结果

两组均正常退出，实际各5步。IDLE：x增量0、观察x增量0；RIGHT：x增量约1.26、观察x增量约0.126。每步观察长度与编码关系检查通过。原生GUI模式启动并取得接口结果，未截图或独立审阅画面，因此GUI视觉现象仍由学习者观看确认。

早期两组敌方子弹数量均为0，只验证了真实空子弹槽路径，不代表非空子弹选取的原生验证完成。护盾比例约0.999700012而非1，因为初始化tick执行了源码shields>=500时的0.15衰减，得到约499.85，再除以500；这不是编码错误。代码中的生命条/护盾比例是状态值，不用它们推断终止。

学习者结果待报告，保持learning。当前证据为原生启动和动作/编码接口检查，不是冒烟训练、策略评测或完整游戏环境验收。下一步根据当前观察边界选定一个受控的训练任务，不从五步成功直接跳到完整DQN。

2026-09-09学习者回复“符合”，确认IDLE/RIGHT实际窗口现象与坐标变化符合要求。081完成；不补写未提供的具体测量值。进入[[082-real-bullet-step-observation]]验证原生非空子弹槽。
