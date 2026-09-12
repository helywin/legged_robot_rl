---
title: 受伤掉盾的提前反馈
status: learning
tags: [强化学习, Chromium, 奖励]
---

## 问题与假设

希望受伤当步就出现负反馈，帮助学习躲避，而不是只等损命。
唯一奖励变化：v5 = v4事件奖励 - 本步shield_damage增量/100。
84维观察、动作、网络和游戏物理不变。100是试验系数，尚未证实最优。
成功条件分开：事件计算及训练接线正确；多种子行为改善才证明策略有效。

## 数据来源与机制

原生HeroAircraft::doDamage在处理伤害后、死亡重生前，累计实际消耗的非负护盾量，
最多计入受伤前可用护盾；不修改物理状态。排除自然衰减、补给、reset、死亡清零。
hello新增shield_damage能力，snapshot.episode_events新增shield_damage累计浮点数。
Python严格要求新版能力/字段并校验有限非负；每步差分进入奖励，
沿现有Transition→经验池→DQN目标→损失→梯度→参数更新链传播。

例：掉28点，原事件奖励0，则新reward=0-28/100=-0.28。
同一步掉40点并拾取道具，奖励为0.2-0.4=-0.2，补盾不抵消受损事件。
掉40点并损命1，无其他事件，则-5-0.4=-5.4。
无盾后的机体伤害本版没有额外逐伤害惩罚，仍由损命惩罚反馈；不同时引入第二个系数。
事件包含碰撞等doDamage调用，不宣称精确归因到子弹。

## 环境与配置

本机CPU，原生无窗口8环境，PyTorch单线程，500次更新冒烟。
原生版本仍chromium-bsu-rl/seeded-reset-v3：物理未改，仅添加观测能力。
以下为实际配置；当前CLI已设每局1000决策，本轮保留，不拿它和旧250上限比较策略效果。

```json
{
  "max_updates": 500,
  "num_envs": 8,
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

仓库根目录：

```bash
third_party/chromium-bsu-rl/scripts/build_chromium_rl.sh
.venv/bin/python -m unittest discover -s third_party/chromium-bsu-rl/tests -p 'test_*.py' -v
.venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -v
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 500 --num-envs 8
.venv/bin/python exercises/chromium_dqn/play.py --check --checkpoint exercises/chromium_dqn/runs/train-4a7eef4faa1643a1bbb3dd8da2a16871/policy.pt
```

## 产物与观察

- 编译日志当时保存在系统临时目录中，文件名为 `chromium-shield-build.log`。
- 训练：exercises/chromium_dqn/runs/train-4a7eef4faa1643a1bbb3dd8da2a16871/。
- 目录含config.json、steps.jsonl、summary.json、policy.pt，不提交生成数据。
- 实际755决策、500更新、总奖励-45.68、last_loss=0.0197104085、参数确有变化。
- 逐行重算755条奖励均一致；2条受伤事件共88点。首条env7、第216决策，
  掉盾28、未死亡、奖励-0.28。这条当时处于预填阶段，仍进入经验池供后续抽样。
- 原生17测试：7通过、10项GUI测试跳过；Python12测试：10通过、2项GUI测试跳过。
- 原生固定seed1..16随机动作验证累计单调、reset清零；
  对无补给无损命且普通护盾低于499的受伤步，累计增量等于实际护盾损失。
- 原生seed31、action0、1tick验证护盾自然衰减但shield_damage=0。
- v5真实权重play --check加载通过；未执行GUI策略回放。

## 护盾归零是否结束

源码：damage初始-500，普通护盾分担80%伤害，同时20%进入机体；
护盾耗尽后机体承担全部伤害。damage>0时损命，lives<0才Game Over。
漏机也会损命；漏机导致最终死亡时主动将护盾清零，不能倒推为掉盾导致死亡。

真实观察：reset(seed=13)，随机动作Random(100013).randrange(18)，每动作5tick；
第238次决策shields=0、damage=-320.1492004394531、lives=0、terminated=False。
这证实无盾仍能活着，lives=0表示还在最后一条命上。

## 结论与下一步

原生计数、奖励接线、短训练及加载通过；没有证明收敛或躲避改善。
后续用相同训练预算、每局上限和种子分别训练v4/v5，再做固定种子评测；
不要拿本次500更新模型与之前40k模型比较算法优劣。
训练默认v5；显式v4参数仍使用旧奖励；旧权重保存的任务版本不会自动变化。
