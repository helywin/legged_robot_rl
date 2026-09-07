# 位置惩罚能否减少CartPole漂移

## 问题

固定100000环境步、100000回放容量，只添加位置惩罚，能否减少小车漂移并延长存活？课程：[[067-position-reward-and-policy]]；原理：[[概念/居中奖励与策略取舍]]。

## 假设

位置扣分比最终出界提供更早反馈，预期位置偏移和出轨比例降低。若扶杆变差或原始步数下降，假设未获支持；不得仅凭加工后的奖励宣称成功。

## 环境与配置

- 证据级别：脚手架静态与短运行接线检查；学习者实际对照待完成。
- Isaac Lab：不适用；任务CartPole-v1，CPU，仓库.venv。
- 单环境，每组100000环境步，从头训练，回放容量100000。
- 网络4→64→ReLU→2，Adam学习率0.001，gamma=0.99。
- batch64，warmup1000，每环境步一次更新，目标同步间隔500次更新。
- epsilon从1降至0.05，下降20000环境步。
- 种子20260904；开发评估20种子20270904～20270923，无探索无更新。
- 基线：当前日志100000步/100000容量平均336.9；正式对照重新运行baseline。
- 唯一实验因素：baseline原始奖励；shaped扣除0.5×(下一位置/2.4)²。

## 命令

```bash
.venv/bin/python experiments/2026-09-07-cartpole-position-reward/train.py baseline
.venv/bin/python experiments/2026-09-07-cartpole-position-reward/train.py shaped
```

## 产物

- 目录：artifacts/cartpole-position-reward/<模式>-<时间戳>/，每次新建。
- 指标：metrics.json（原始步数），reward-experiment.json（参数、结束类型、位置/角度均方根）。
- 检查点：online-network.pt；导出：online-network.onnx及可能的.onnx.data，拷贝模型时保持同目录。
- UI：`.venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/watch.py <本轮online-network.pt路径>`。
- 不提交模型或生成数据；没有生成视频。

## 观察与指标

学习者结果待记录。比较原始平均步数、纯时间到达上限回合、位置越界、杆角越界和位置/杆角均方根。同一步可同时违反两类边界；达到500步的计数应结合结束类型阅读。

## 结论

尚未运行完整奖励对照，未证明位置惩罚能改善当前策略。课程保持learning。

## 未验证与下一步

下一问题：位置更居中时，杆角和存活步数是否一起改善？先判断此项取舍，再决定是否调整位置权重。不在同一轮叠加速度、角度或动作切换惩罚。
