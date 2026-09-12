# Chromium DQN 恒定动作故障、稳定性审计与五动作接入

## 问题与假设

用户百万更新模型从开局冲向角落。检查是否为输入、动作或训练更新错误，再单独比较损失函数。随后按用户要求接入五个移动选择并固定开火。

## 原始故障证据

检查点`exercises/chromium_dqn/runs/train-335f5b5837a74581aa6a82a2e878ef39/policy.pt`：v7，36维，18动作，MSE，24环境，1000000次更新，seed=1真实无窗口回放177次决策，全部action=15，第5次到达右上角(10,7.5)。各观察得到完全相同的18项Q值；两层ReLU隐藏层全零，输出只剩最后一层偏置。首步最大Q约3.116815，实际折扣回报约-6.375455。全局无子弹命中/击杀/拾取，5次漏机、5次损命。

历史日志最大loss=108815.4375（更新54829），最终loss很小不能证明学到策略。没有历史中间权重，不能确定失活发生时间，也不能仅凭这些证据证明唯一原因。追踪数据当时保存在系统临时目录中，文件名为 `chromium-v7-seed1-trace.json`；它不是仓库内的持久产物。

## 审计范围与修复

检查独立项目runtime、task、replay/tensor_replay、network、agent/fast_update、train、evaluate、play及其原生游戏接口，未将历史全部课程纳入重写范围。

- 更新目标：自然结束禁止估计未来，外部截断保留同局末观察，不混入reset观察；目标网络不反向求导。
- 并行采样：全局预算、独立环境种子、动作与同局经验对应；共享模型按经验更新。
- 原生动作：holdFire只在状态变化时触发fireGun，不会每tick重新启动射击冷却；原地松开方向仍有游戏惯性。
- 事件：实际子弹伤害、击殺、护盾损失与漏机事件走同一任务奖励；无新原生代码修改。
- 修复经验插入形状检查，防止长度1观察被广播；回放/评测明确指定检查点，消除过时默认路径；统一直接调用与CLI的每局上限为1000。
- 训练默认Huber替换MSE；大误差时损失对误差的梯度受限，但不保证网络所有参数梯度有界。保留`--loss-kind mse`作对照。
- 增加非有限loss拒绝更新、每1000更新/最终更新检查隐藏层与Q值、失活保存诊断并停止、每10000更新推理权重及源码SHA256记录。检查256条真实经验，只能发现指定形式的退化，不能证明策略正确。

## 损失单因素实验

环境：CPU、无窗口真实Chromium，原生版本seeded-reset-v3，v7，N=24，各100000次更新；gamma=.99、lr=.001、经验10000、batch32、预填256、目标同步100、单局1000决策。种子game31/network7/replay7/exploration11。epsilon按本次100000预算衰减，所以不是原百万运行的前100000步。两组部分并发执行，耗时不能作为吞吐对照。

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 100000 --num-envs 24 --task-version task-v7-hit-feedback --loss-kind mse
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 100000 --num-envs 24 --task-version task-v7-hit-feedback --loss-kind huber
```

产物目录分别为`runs/train-4ba3e45e2d9943aea6ee3794635c8e2b`和`runs/train-3b4faff3e9684803a8701463b470dcdd`（相对exercises/chromium_dqn）。诊断使用`diagnose.py --checkpoint <对应目录>/policy.pt --seed 1`；评测使用`evaluate.py --run --checkpoint <对应目录>/policy.pt`，固定种子30001..30020，另有相同种子随机基线。

seed1：MSE存活177决策，子弹击杀0，子弹伤害10.5；Huber存活207决策，子弹击杀1，子弹伤害117，实际选择过15种动作。此处15是使用过的动作数量，并非动作空间大小。两组100000更新末的网络均未全零。

固定20局：MSE平均子弹击杀0.3、伤害67.475；Huber平均子弹击杀0.05、伤害30.9；随机平均子弹击杀0.25、伤害87.875。全部死亡、无通关。Huber的单局改善不代表整体优于MSE或随机，更没有证明长期收敛。

## 用户指定五动作

新增task-v8-five-actions并设为训练默认：0原地、1上、2下、3左、4右，全程开火，原生映射9..13。观察及奖励与v7完全相同。网络输出5项，探索、评测随机基线、诊断和GUI回放按任务版本使用同一映射；旧18动作权重保持旧解释。高级武器弹药仍会被自动开火消耗。

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 10000 --num-envs 8
```

五动作短训练产物`runs/train-dcf185e4d7254dc9b53eaea39d2ced0c`：10000次更新，10255决策，48次结束，参数发生变化。10条health记录全部有限且未全零，已生成最终和10000更新推理权重。此运行是接线验证，不与上述100000更新组比较策略优劣。

## 验证与边界

```bash
RUN_CHROMIUM_GUI_TESTS=1 .venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -v
RUN_CHROMIUM_GUI_TESTS=1 .venv/bin/python -m unittest discover -s third_party/chromium-bsu-rl/tests -p 'test_*.py' -v
```

原生18项通过（含GUI）；独立项目27项通过（含GUI、五动作与原生固定开火的逐步真实轨迹一致性、Huber编译/普通更新等价、失活检查、错误观察形状）。旧18动作和新5动作中间权重通过play加载检查。另人工注入死亡隐藏层，真实运行1000更新后正确停止并保存diagnostic.pt/failure.json，未保存正常policy.pt：`runs/train-ff37f646da7a4847a17c539b6ad979f9`。

证据层级：源码检查、真实游戏接口/GUI回归、短训练与冻结策略评测。没有通关证据，不涉及机器人真机。

## 结论与下一个问题

确认旧策略网络已失活，不应继续把贴角落只解释成奖励不足。稳定性检测与五动作已接入；尚未证明策略能收敛。下一组受控比较只改变动作空间：v7对v8，两组均Huber、相同预算和种子，并比较固定种子命中/击杀/漏机与死亡分布，不能只挑一局看画面。

五动作短训练后冻结评测亦完成：模型/随机各20局，模型通关率0.0，死亡率1.0。seed1诊断动作编号全部在0..4，参数未改变；评测目录`runs/eval-e4fdc98955914b33a9b9a7e886370477`，诊断目录`runs/diagnose-b53a879d51ab46fd81a1ae4a0e4ca076`。仍不能宣称已收敛。
