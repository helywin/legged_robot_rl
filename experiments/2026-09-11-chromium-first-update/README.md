# 重现首个真实DQN更新

问题：原训练第一次loss=19.86665主要由什么经验贡献，梯度如何改变参数？
假设：高奖励样本主导初始误差，但更新应只影响在线网络。
环境：现有CPU PyTorch、Chromium seeded-reset-v3、task-v1；使用学习者当前Agent/Task/Replay/训练循环，未复制旧实现。
配置：读取首轮train-06fbbc3a5a1c4ff78afc7ffabb1586c1/config.json，唯一预算变化为max_updates=1；其余配置与种子相同（游戏31，网络7，回放7，探索11）。此变化停止于首个更新，不改变此前调度。
命令（仓库根）：

```bash
.venv/bin/python experiments/2026-09-11-chromium-first-update/inspect_update.py
```

教师执行256次决策、1次更新。诊断调用学习者的Agent.update，捕获实际前向计算图与输出梯度；统计用额外no_grad前向，不改变参数或随机状态。没有覆盖原检查点。产物为exercises/chromium_dqn/runs/first-update-diagnostic.json。

结果：

```json
{
  "batch_size": 32,
  "row": 14,
  "action": 14,
  "reward": 25.0,
  "terminated": false,
  "truncated": false,
  "prediction": -0.06658826768398285,
  "next_max_q": 0.1143907681107521,
  "gamma": 0.99,
  "target": 25.11324691772461,
  "error": -25.179834365844727,
  "squared_error": 634.0240478515625,
  "squared_error_sum": 635.7327880859375,
  "loss": 19.866649627685547,
  "selected_q_gradient": -1.5737396478652954,
  "analytic_selected_q_gradient": -1.5737396478652954,
  "selected_bias_before": -0.0663372278213501,
  "selected_bias_gradient": -1.5913128852844238,
  "same_action_rows": 3,
  "learning_rate": 0.001,
  "selected_bias_after": -0.06474591791629791,
  "q_graph_node": "AddmmBackward0",
  "target_unchanged": true,
  "target_grad_none": true,
  "next_forward_prediction": -0.06370468437671661,
  "decisions": 256,
  "original_first_loss": 19.866649627685547,
  "first_loss_exact_match": true
}
```

结论：首次loss与原日志精确一致；第14行（零起始）奖励25，占整批平方误差约99.7%；该选中Q值的梯度与2×误差/32一致。相同动作共有3条经验，输出偏置梯度是这些行贡献的总和，不能等同单行Q梯度。SGD按学习率0.001更新在线参数，目标参数不变且没有梯度。

验证边界：这是当前构建/配置/种子下对首个更新的重现，不是保存了原训练全过程状态，不声称整段500次更新已逐值重现，也不证明策略优于随机。
下一个问题：冻结策略在多个独立开局里是否优于同规则随机动作？对应092。
