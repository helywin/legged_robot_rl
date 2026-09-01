# CartPole 随机策略基线

## 问题

Gymnasium 的 `CartPole-v1` 是否能在仓库 `.venv` 中按标准接口启动？不学习的随机策略在固定 100 个种子上能坚持多久？

## 假设

`reset()` 应返回四项观察和 `info`，`step()` 应返回新观察、奖励、`terminated`、`truncated` 和 `info`。随机动作没有根据杆状态纠正方向，预计会很快自然终止，远低于 500 步上限。

## 环境与配置

- 证据级别：启动检查与随机策略基线，不是训练
- Python：3.14.6，仓库 `.venv`
- Gymnasium：1.3.0
- NumPy：2.5.2
- pygame-ce：2.5.8，本次无界面运行
- Isaac Lab 版本或提交：不适用
- 任务名称：`CartPole-v1`
- 并行环境数：1
- 训练迭代数：0
- 回合数：100
- 随机种子：基础种子 `20260901`，第 N 轮使用 `20260901 + N`
- 基线：每步使用 `env.action_space.sample()` 随机选左右动作
- 只改变的变量：相对之前的纯 Python FrozenLake，只把环境换成 Gymnasium CartPole；策略仍不学习

## 命令

```bash
.venv/bin/python -m pip install --editable '.[dev]'
.venv/bin/python -m examples.cartpole_random_baseline \
  --episodes 100 \
  --seed 20260901
```

## 产物

- 脚本：`examples/cartpole_random_baseline.py`
- 测试：`tests/test_cartpole_random_baseline.py`
- 日志位置：终端标准输出，未生成独立日志文件
- 检查点位置：无；没有训练模型
- 图表或视频位置：无；本次无界面运行

## 观察与指标

第一步接口输出：

```text
reset_observation=[0.013218, 0.008241, -0.014578, -0.020939]
action=1 (向右推)
reward=1.0
terminated=False
truncated=False
```

100 回合统计：

```text
mean_return=21.40
shortest_episode=9
longest_episode=57
terminated_episodes=100
truncated_episodes=0
```

由于默认奖励每步为 `+1`，回报与回合步数相同。固定种子重复运行得到相同汇总。

## 结论

数据支持假设：真实环境接口成功启动；随机策略平均只坚持约 21 步，100 轮全部在达到 500 步上限前自然终止。它可以作为后续 DQN 训练的对照基线。

结论只属于 CartPole 启动检查和随机策略运行证据，不是 DQN 冒烟训练或独立策略评测。

## 未验证与下一步

没有验证神经网络前向预测、经验回放、DQN 参数更新、检查点保存或 GUI 回放，也不能推出 Chromium B.S.U. 或 Go2 的任何结果。

下一步只建立接收四项观察、输出两个动作 Q 值的最小 CartPole Q 网络，并检查输入输出形状，不开始完整训练。
