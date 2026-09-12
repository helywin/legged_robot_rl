---
title: 36维观察与近期平均奖励
status: learning
tags: [强化学习, Chromium, 观察]
---

## 问题与假设

用户要求直接缩减至36维，以检验较少对象是否更易学习；本轮只做实现及短训练接线验证。
唯一观察变化：从v5的自身8+敌机4×3+敌弹8×3+道具4×10，
减少为自身8+敌机2×3+敌弹4×3+道具1×10=36。
保留已有缩放、排序、空槽、字段、18动作和64→64隐藏层。
奖励完全沿用当前v5，包括用户修改的击毁×10和受伤掉盾惩罚。
不拿用户之前v4百万更新作为同奖励的严格对照。

## 接口

新版本task-v6-36；训练TrainConfig与CLI默认均选它。
网络输入、经验池、checkpoint元数据、评测和回放按任务版本自动选择36维。
旧版维数映射保留，已有84维权重不自动变成36维。

进度条另显示最近1000条经验的平均reward，多个环境按已有记录顺序合并。
包含预填阶段，窗口不足时用实际条数；显示节流不丢样本。
这是每次决策的奖励均值，不是episode总奖励或原始游戏得分。
额外显示不改变训练/奖励计算。

## 环境与配置

CPU、无窗口、当前原生shield_damage接口；无C++修改。实际短训练配置：

```json
{
  "max_updates": 2000,
  "update_backend": "eager",
  "num_envs": 8,
  "task_version": "task-v6-36",
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

## 命令

```bash
.venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -p 'test_small_observation.py' -v
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 2000 --num-envs 8
.venv/bin/python exercises/chromium_dqn/play.py --check --checkpoint exercises/chromium_dqn/runs/train-a4d1a7904b1240e9be0699fea147c74e/policy.pt
.venv/bin/python exercises/chromium_dqn/evaluate.py --run --checkpoint exercises/chromium_dqn/runs/train-a4d1a7904b1240e9be0699fea147c74e/policy.pt
```

## 产物与结果

训练产物：exercises/chromium_dqn/runs/train-a4d1a7904b1240e9be0699fea147c74e，含config.json、steps.jsonl、summary.json、policy.pt。
本轮之前另完成500更新入口冒烟；此处为2000更新含均奖显示的运行：

```json
{
  "decisions": 2255,
  "episodes_finished": 8,
  "updates": 2000,
  "total_reward": -221.96597427368184,
  "last_loss": 0.7611898183822632,
  "parameter_changed": true,
  "elapsed_seconds": 1.1470607530209236,
  "native_version": "chromium-bsu-rl/seeded-reset-v3",
  "num_envs": 8,
  "headless": true,
  "update_backend": "eager",
  "backend_init_seconds": 8.79520084708929e-05,
  "updates_per_second": 1743.5868106660937,
  "decisions_per_second": 1965.8941290260207
}
```

- 2项新观察检查通过：维数、槽位范围、道具选择、空槽和当前v5/v6奖励一致。
- play --check验证真实36维权重加载。
- 人工1200条递增奖励测试验证滚动均值：最后1000条200..1199，均值699.5。
- 实际日志最后1000条reward重算与进度条显示一致。
- 冻结模型/随机各20局评测接线，种子30001..30020，参数不更新。
- 评测目录：exercises/chromium_dqn/runs/eval-5b3a57ff3803410f90136592ce9e6e20。

## 结论与下一步

实现、原生短训练和评测接口验证通过；未做GUI回放。
本轮预算仅用于接线，不能说明已收敛或36维一定比84维好。
后续相同奖励/种子/预算比较v5与v6，并保存中间权重观察是否后期退化。
