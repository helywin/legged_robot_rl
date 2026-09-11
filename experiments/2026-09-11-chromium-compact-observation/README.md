---
title: 去掉运动字段的84维观察对照
status: learning
tags: [强化学习, Chromium, 实验]
---

## 问题与假设

删掉运动信息、缩小输入，是否能在相同更新预算下更早学会攻击和生存？
假设：这些字段对当前任务的帮助可能小于学习负担，删除后可能改善行为；也可能因信息不足退化。
成功条件：在多个训练种子、多个更新预算下稳定改善独立评测表现；本次先做单种子试验，不以loss下降判成功。

## 唯一变化量与规格

对照task-v3-events 110维，候选task-v4-no-motion 84维。
删除玩家键盘运动量2项、8颗敌弹速度16项、4件道具下一tick预计位移8项，共26项。
敌机原本不输入速度。保留自身8、敌机4×3、敌弹8×3、道具4×10。
观察函数从原编码投影，因此缩放、排序、容量和空槽不变；原生接口仍传输完整数据。
这不是传输协议压缩，不能据此承诺整条采样链路加速。

两版事件奖励完全一样：击毁+1、拾取+0.2、损命-5、过关+20。
隐藏层均64→64，输出18动作；输入层随维数改变，总参数12434→10770（减少约13.4%）。
相同随机种子不代表两种形状的网络初始参数逐项相同；这是本次观察设计变更伴随的限制。

## 环境与配置

本机CPU、8个真实原生游戏进程、无窗口、PyTorch单线程。
原生行为版本chromium-bsu-rl/seeded-reset-v3；子仓库提交6b5d0662dc12b64649b049a41a521ca55e7cd6b8。
本次没有修改C++。代码基于主仓库0d3af374a125bbd6b21bd4d7acfd06e3334a1e1b。
仅任务版本不同，其他训练配置逐字段核对一致。4k组仅max_updates改为4000：

```json
{
  "max_updates": 40000,
  "num_envs": 8,
  "task_version": "task-v4-no-motion",
  "episode_limit": 250,
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

训练开局seed=31+环境编号+该环境已结束局数×8；评测为30001..30020。
每个模型评测20局，同时运行相同种子的随机策略20局；不更新参数。
两组预算分别从头训练，非续训；只有一组网络/探索/经验池随机种子。
110维训练复用本轮之前已有同配置产物；40k基线用当前代码重新评测，全部汇总与旧结果完全一致。

## 完整命令

仓库根目录执行；分别产生4组训练产物，随后评测每组终端打印的准确路径：

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 4000 --num-envs 8 --task-version task-v3-events
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 4000 --num-envs 8 --task-version task-v4-no-motion
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 40000 --num-envs 8 --task-version task-v3-events
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 40000 --num-envs 8 --task-version task-v4-no-motion
```

本次候选与基线复测的准确评测命令：

```bash
.venv/bin/python exercises/chromium_dqn/evaluate.py --run --seed-start 30001 --checkpoint exercises/chromium_dqn/runs/train-6274dbb0be344d6caaaa3dd760c67cf9/policy.pt
.venv/bin/python exercises/chromium_dqn/evaluate.py --run --seed-start 30001 --checkpoint exercises/chromium_dqn/runs/train-c50855a8f75140ff9aeb160b88dfbd9a/policy.pt
.venv/bin/python exercises/chromium_dqn/evaluate.py --run --seed-start 30001 --checkpoint exercises/chromium_dqn/runs/train-78de10fce1bf4d82a46a067b365fd6a0/policy.pt
```

## 产物与结果

以下目录均位于exercises/chromium_dqn/runs/，不提交权重或逐步日志。
训练目录含config.json、steps.jsonl、summary.json、policy.pt；
评测目录含config.json、episodes.jsonl、summary.json，配置保存源码与权重哈希。

- 110维、4000更新：训练 `train-d78a52f1fcae410da8a3be20f68981d9`；评测 `eval-e1f9b88a6a3148998351b1869b972447`。
- 84维、4000更新：训练 `train-6274dbb0be344d6caaaa3dd760c67cf9`；评测 `eval-b4d76b167edd4278803b721db76576f4`。
- 110维、40000更新：训练 `train-78de10fce1bf4d82a46a067b365fd6a0`；评测 `eval-4e1afb38f33c49288c047cff93949db1`。
- 84维、40000更新：训练 `train-c50855a8f75140ff9aeb160b88dfbd9a`；评测 `eval-6e1728183a4a49fe89d54ad7db2eebee`。

|维数|更新数|平均击毁|平均拾取|平均存活秒|死亡/20|截断/20|训练秒|
|---|---|---|---|---|---|---|---|
|110|4000|0.70|0.40|19.488|19|1|2.246|
|84|4000|0.50|0.35|19.305|20|0|2.297|
|110|40000|0.15|0.35|18.595|20|0|21.069|
|84|40000|1.95|0.15|19.903|17|3|21.005|

全部模型过关0局；截断仅表示达到250决策上限。
随机策略平均击毁0.85、拾取0.95、存活19.185秒，20局全死亡。
84维40k原始平均得分7336.25，其中7250来自漏接道具得分，因此高分不能用作成功证据。

## 结论与验证边界

40k压缩版击毁和存活有改善，拾取下降；4k没有同样改善。
训练耗时21.005秒，对照21.069秒，单次差异不足以说明吞吐提升。
目前不支持“已经更快收敛”的结论，只支持保留84维作为后续候选。默认仍为v3，显式参数选择v4。

测试11项：9项通过、2项需GUI而跳过；check_task通过。
v1/v2/v3/v4各自真实权重的play --check通过。
已完成真实无窗口训练与固定种子评测，评测确认参数不变。
没有执行GUI行为观察，不能宣称视觉上的贴角落已解决；更不涉及机器人/真机验证。
课程保持learning，教师运行不替代学习者理解或完成能力验收。

## 下一个问题

保持奖励和训练配置，换几组训练随机种子，确认40k的击毁改善能否复现。
不要同时加位置惩罚、改奖励或更换算法，否则无法判断改善来自哪里。
