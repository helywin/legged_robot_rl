---
title: 子弹逐次命中奖励
status: learning
tags: [强化学习, Chromium, 奖励]
---

## 问题与假设

当前奖励把所有enemies_destroyed都按×10奖励，包含碰撞、重生爆炸，
可能出现损命-5、清掉2敌机+20的正净收益。源码确认此风险，未证明模型实际利用。
用户选择用命中反馈设计，强调不能只在最终击毁时给一次奖励。
本轮不加对准x奖励，不奖励空开火或每步存活；36维观察与其他奖励保持不变。
假设：提前到每次实际削血反馈，并排除非射击清场，能改善攻击学习；待长训练验证。

## 最终v7规则

每步奖励：
子弹实际削血比例之和 + 2×子弹完成击毁数 + 0.2×拾取数
- 5×实际损命数 - 实际受伤掉盾量/100 + 20×首次过关。

每次命中收益=实际削减的敌机血量/该敌机初始血量。
每次实际削血都给分，持续伤害武器逐tick累计；每个决策汇总其执行的tick。
截断到剩余血量，已经死亡敌机不再贡献削血，过量伤害不给分。
子弹使damage从<=0跨越到>0时计击毁一次，随后清理不重复计数。
碰撞、重生爆炸、漏机不计子弹伤害/击毁，原总击毁计数保留为诊断信息。
子弹自己触发的连锁爆炸未追溯归因，只有直接子弹伤害获得本版攻击收益。

例：100血敌机依次被子弹打掉20、30、50，伤害奖励0.2、0.3、0.5；
最后击毁另+2，总计3。不是一次性等待击毁。
没命中、没其他事件的步骤仍可为0；不人为制造每步正奖励。
当前未增加机体伤害惩罚，护盾耗尽后的损失仍通过损命反馈。

## 原生与Python接口

HeroAmmo.cpp命中伤害更新处采集来源明确的事件：
projectile_damage、projectile_damage_fraction、projectile_kills。
全局累计reset归零，hello声明projectile_damage能力。Python严格校验字段，
缺失不默认为0。事件差分进入Transition和原有DQN更新。
仅增加计数，不修改游戏伤害、物理或随机数。

新任务task-v7-hit-feedback，训练默认v7、36维；旧任务按已有规则保留。
v3..v6保留学习者之前的总击毁×10；同步两项仍按×1预期的旧测试。
epsilon保持预算前80%衰减、后20%低探索；gamma和SGD不变。
推理权重按任务版本加载，不将旧权重自动迁移成新奖励学到的策略。

## 配置与完整命令

实际4000更新、8个CPU无窗口环境配置：

```json
{
  "max_updates": 4000,
  "update_backend": "eager",
  "num_envs": 8,
  "task_version": "task-v7-hit-feedback",
  "episode_limit": 1000,
  "capacity": 10000,
  "batch_size": 32,
  "learning_starts": 256,
  "target_sync_every": 100,
  "epsilon_start": 1.0,
  "epsilon_end": 0.05,
  "epsilon_decay_fraction": 0.8,
  "gamma": 0.99,
  "learning_rate": 0.001,
  "game_seed": 31,
  "network_seed": 7,
  "replay_seed": 7,
  "exploration_seed": 11
}
```

仓库根目录：

```bash
third_party/chromium-bsu-rl/scripts/build_chromium_rl.sh
.venv/bin/python -m unittest discover -s third_party/chromium-bsu-rl/tests -p 'test_*.py' -v
.venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -v
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 4000 --num-envs 8
.venv/bin/python exercises/chromium_dqn/evaluate.py --run --checkpoint exercises/chromium_dqn/runs/train-d26211ac9e2743e3bc5ccd4bd6713459/policy.pt
```

## 验证与结果

- 原生18项：8通过、10项GUI测试跳过；Python21项：19通过、2项GUI测试跳过。
- 原生真实seed1..16验证：未致死命中能增加伤害计数；不开火时即使有总击毁，
  三个射击计数仍全部为0；reset与累计单调测试通过。
- Python分三次模拟削血，奖励逐次0.2、0.3、2.5；损命并爆炸清2机仍为-5。
- 真实训练4255决策、4000更新，逐条重算全部奖励一致。
- 200条命中未击毁经验有伤害反馈。第33决策掉敌机7血，奖励0.063636；
  第65决策再掉3.5血，奖励0.031818；并非击毁后才给一次分。
- 真实权重play --check通过；冻结模型/随机各20局评测运行完成，参数保持不变。
- 训练产物：exercises/chromium_dqn/runs/train-d26211ac9e2743e3bc5ccd4bd6713459。
- 评测产物：exercises/chromium_dqn/runs/eval-8a8aad0b153e4faa86ea2be7c0a3b642。
- 当时的系统临时目录中保存了编译日志 `chromium-hit-build.log` 和测试日志 `chromium-hit-*-tests.log`。
- 生成权重和逐步日志不提交。

评测汇总（仅短训练接线证据，不与旧版百万更新作效果对照）：

```json
{
  "model": {
    "episodes": 20,
    "mean_score": 7210.0,
    "median_score": 5000.0,
    "mean_seconds": 19.611,
    "mean_life_decreases": 5.05,
    "completion_rate": 0.0,
    "death_rate": 1.0,
    "truncation_rate": 0.0,
    "mean_events": {
      "enemies_destroyed": 0.8,
      "enemies_escaped": 4.85,
      "lives_lost": 5.05,
      "pickups": 0.35,
      "missed_powerups": 1.95,
      "pickup_score": 35.0,
      "missed_powerup_score": 7125.0,
      "shield_damage": 310.53990944381803,
      "projectile_kills": 0.3,
      "projectile_damage": 82.225,
      "projectile_damage_fraction": 0.9050757569260895
    }
  },
  "random": {
    "episodes": 20,
    "mean_score": 5351.25,
    "median_score": 5000.0,
    "mean_seconds": 19.185000000000002,
    "mean_life_decreases": 5.05,
    "completion_rate": 0.0,
    "death_rate": 1.0,
    "truncation_rate": 0.0,
    "mean_events": {
      "enemies_destroyed": 0.85,
      "enemies_escaped": 5.05,
      "lives_lost": 5.05,
      "pickups": 0.95,
      "missed_powerups": 1.5,
      "pickup_score": 65.0,
      "missed_powerup_score": 5250.0,
      "shield_damage": 180.48986129760743,
      "projectile_kills": 0.25,
      "projectile_damage": 87.875,
      "projectile_damage_fraction": 1.1205808248370885
    }
  },
  "elapsed_seconds": 1.2200073930434883,
  "parameters_unchanged": true
}
```

## 结论与下一步

逐次伤害奖励及来源区分已由原生运行验证；不是已收敛或已通关。
未做GUI行为观察。下一步固定种子、观察和探索日程，以相同预算比较旧/新奖励，
并在训练中途评测，检查是否再次退化。若命中仍稀疏，再单独研究弱对准引导，
不在本版混入位置或空开火奖励。
