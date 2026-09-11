# Chromium DQN 独立实作项目

状态：独立DQN训练、无窗口并行游戏、评测和GUI回放均已实现。当前默认task-v8-five-actions：36维观察、5动作固定开火，使用Huber损失；策略效果尚未达到通关要求。旧课程说明保留为历史记录。

本项目的隔离指代码、配置与运行产物隔离；仍使用仓库`.venv/bin/python`及根目录`pyproject.toml`管理依赖。外部依赖只有通用库和C++游戏可执行程序。沿用游戏协议，但重新实现本项目的Python通信层，不依赖旧`chromium_rl`客户端。不重写游戏本体。

## 四层职责

```text
train / evaluate / play       组织流程，管理运行与产物
          ↓
     task  +  dqn            任务定义与学习算法，各自独立
       ↓
     runtime                 游戏进程、协议通信与原始快照
       ↓
   C++ Chromium B.S.U.       游戏真实状态与时间推进
```

算法只接受数值观察和经验，不读取游戏对象；通信层只交付原始游戏事实，不计算学习奖励。训练、评测、GUI回放使用同一套任务规则与动作映射。

## 计划目录及实现责任

以下是待实现的文件设计，不表示已经存在或可以运行。按工作段创建，避免先堆空模块。

| 文件 | 职责 | 主要实现者 |
|---|---|---|
| `runtime.py` | 启动/关闭进程、请求超时、协议校验、reset/step/render、原始状态解析 | 教师重新实现，并验证C++配套能力 |
| `task.py` | 观察编码、动作映射、奖励、回合上限、任务reset/step | 学习者 |
| `replay.py` | 有界经验存储、独立随机采样 | 学习者 |
| `network.py` | 输入维数到每个动作Q值的网络 | 学习者 |
| `agent.py` | 探索与决策、批量更新、目标网络同步 | 学习者 |
| `train.py` | 环境交互、经验入库、更新节奏、跨回合与保存 | 学习者 |
| `evaluate.py` | 冻结模型/随机基线、逐回合统计 | 学习者，在092完善 |
| `play.py` | 加载策略，使用同一任务与GUI观察行为 | 学习者 |
| `config.py`、`configs/first_level.json` | 配置读取、合法性检查、首版明确数值 | 教师提供样板，任务取值讲解后共同确定 |
| `artifacts.py` | 运行目录、配置快照、日志与检查点文件读写 | 教师提供通用文件样板；学习状态内容由学习者连接 |
| `tests/` | 游戏边界、经验边界与更新行为的有意义验证 | 教师组织，结合学习者调试 |

首版用直接函数调用和少量数据类表达返回值，不引入插件注册、通用算法基类、事件总线或分布式训练。数据类只是把有名字的字段放在一起，开始实现时再解释具体语法。

## 接口约定：返回值不再靠猜

这里是设计契约，不是现有游戏API。原生能力缺失时由教师先补齐并验证，不能用假数据实现成功返回。

| 接口 | 输入 | 返回什么 |
|---|---|---|
| `Runtime.reset(seed)` | 游戏随机种子整数 | `RawSnapshot`：原始开局状态；reset内部完成后才返回 |
| `Runtime.step(action_id, ticks)` | 合法游戏动作编号、正整数tick数 | `RawStep`：本步后的快照、实际推进tick数、游戏终止标志及原因 |
| `Runtime.render()` | 无 | 无返回值；显示当前状态，不推进游戏 |
| `Runtime.close()` | 无 | 无返回值；回收本实例资源，可重复调用 |
| `Task.reset(seed)` | 游戏种子 | `ResetResult(observation, info)`；观察为固定长度浮点列表 |
| `Task.step(action)` | 策略动作索引 | `StepResult(observation, reward, terminated, truncated, info)` |
| `Replay.add(transition)` | 一条完整经验 | 无返回值 |
| `Replay.sample(batch_size)` | 正整数样本数，不超过现有数量 | `list[Transition]`；一次采样不重复选同一存储位置 |
| `Agent.act(observation, epsilon)` | 一条观察、探索概率 | 一个合法整数动作索引 |
| `Agent.update(transitions)` | 一批经验 | `UpdateStats`：损失、平均预测/目标、累计更新次数等诊断值 |

`Transition`明确包含：`observation, action, reward, next_observation, terminated, truncated`。`next_observation`必须是执行动作后的同局观察，不能填自动重开后的开局状态。所有存入经验的观察独立保存，后续状态改变不能覆盖旧经验。

`info`只放诊断数据，例如原始分数、损命、关卡、实际ticks和结束原因，不自动拼进策略输入。初版观察维数、单位、容量、填充及缩放规则在任务实现前冻结；不能直接沿用旧17项的假设。

游戏自然结束设置`terminated`；外部时间/决策上限设置`truncated`。训练总预算耗尽只结束本次采集，不伪造环境终止。首版外部上限视为采集截断，更新时仍可使用结束前下一观察估计未来价值；游戏自然终止不再估计未来。两者同时发生时按自然终止处理目标。先完整解释这些计算，再让学习者实现。

任务结束后必须显式reset，step不能偷偷重开。进程退出或通信超时属于运行错误，不能伪装成死亡、奖励零或有效经验。

## 配置与产物

配置分成五组：runtime（程序路径/超时/绘图）、task（观察/动作/ticks/奖励/上限）、agent（网络/折扣/优化参数）、training（容量/批量/探索/同步/预算）、evaluation（回合数/开局种子）。先用JSON和标准库，不增加配置框架。

任务规格携带版本、观察字段顺序与维数、动作映射和奖励定义。检查点保存该规格；加载时不兼容就说明差异并拒绝运行，不能凭相同维数直接使用。

每次运行写入本目录忽略的`runs/<run_id>/`：冻结配置、代码版本及脏状态、原生版本、各随机种子、逐回合统计、逐更新统计、检查点。可复现命令和结论另记入仓库`experiments/`；不提交模型和大日志。

区分游戏种子、探索种子、回放采样种子及网络随机状态。评测使用独立游戏实例和随机数流，不改变训练回放或探索状态。续训至少恢复网络、目标网络、优化器、计数、探索进度、回放和Python/网络随机状态；没有原生状态快照时仅承诺从新回合继续，不能声称逐步无缝恢复。

## 实现节奏

090在同一项目内分三段推进，每段先讲清机制，再提供自包含练习说明和核心TODO：

1. **完成任务闭环。** 教师先交付经原生验证的runtime；学习者从零写task，运行完整一局并正确重开。
2. **完成一次真实学习更新。** 学习者从零写replay、network和agent；用一批明确经验解释并验证参数为何改变。
3. **完成短训练与回放。** 学习者写train/play并连接保存加载；按090预算完成非零更新、跨回合和GUI回放。

这三段不另开碎片课程，也不要求所有模块一次写完。091解释本次真实更新，092实现完整评测，093诊断，094单因素修改，095重复验证与续训，096扩展关卡，097阶段验收。

教师提供的每个练习文件必须写清场景、接口、修改位置、运行命令与成功条件；提供脚手架时不提前填核心答案。当前没有训练入口。已有通信演示和Task练习检查命令见下方。

## 验证边界

- 静态检查确认层间依赖，禁止导入旧`examples`、兄弟练习及旧`chromium_rl`客户端，也禁止复制其实现。
- Python验证覆盖经验不被覆盖、回合边界、动作/Q值形状、目标网络与参数更新等真实错误风险。
- 原生验证覆盖reset/seed、动作释放与开火、ticks/render、首版观察、真实得分/损命/终止、连续开局及资源清理。模拟通信测试不能代替这些证据。
- 短训练验证闭环，独立评测判断策略能力，两者分别记录。

当前实现和证据以runtime-readiness.md为准；Task及DQN仍未完成。

## 当前动手入口

从仓库根目录运行：

```bash
.venv/bin/python exercises/chromium_dqn/runtime.py
.venv/bin/python exercises/chromium_dqn/check_task.py
# 四个TODO完成且逻辑检查通过后，运行真实游戏：
.venv/bin/python exercises/chromium_dqn/check_task.py --native
```

`task.py`顶部包含完整题目、首版62项观察与奖励规格、返回值和错误处理要求。四个TODO由学习者实现，教师检查器不含答案。`runtime.py`是教师底层工程演示，不算学习者完成训练。

新runtime检查：

```bash
RUN_CHROMIUM_GUI_TESTS=1 .venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -v
```

## 第二工作段：一次真正的学习更新

Task完成后，当前动手入口为`network.py`和`agent.py`；核心TODO由学习者实现，完整输入输出与手算示例在文件顶部。运行：

```bash
.venv/bin/python exercises/chromium_dqn/check_update.py
```

检查器注入固定两条经验，验证真实参数更新，不提供替代答案。当前脚手架尚未完成，不能称为已开始游戏训练。

## 当前入口：真实训练循环

回放、动作选择和批量更新连接已通过学习者实现检查。继续实现`train.py`中的`train_loop`，其余启动/日志/保存样板已提供。先运行`train.py --check`，通过后再运行`train.py --run`。当前循环未实现，无真实训练结果；保存的policy.pt设计为推理权重，尚不是续训快照。

## 当前进度：真实训练已完成，准备GUI回放

学习者首轮755次决策、500次更新，已保存policy.pt并核对；详情见仓库experiments/2026-09-11-chromium-first-dqn。当前实现play.py中的play_episode；先用`play.py --check`验证加载，再用`play.py --run`观看策略。核心回放尚未实现，尚无策略有效性评测。

## 当前入口：092模型与随机策略评测

090真实训练与修复后的回放完成，091真实更新复盘结束。现在实现evaluate.py的两个TODO，先--check再--run，执行同任务/同种子列表的模型和随机策略各20局比较。暂无评测结果，不预判模型获胜。


## 最新进度：093失败复盘

092学习者运行的40局已核对：模型/随机平均分6607.5/4832.5，中位5087.5/5000；模型19局死亡、1局达上限，双方0过关。此前各工作段“尚未实现”描述为当时状态，当前Task、训练、回放和评测都已完成。

现在无需新增函数，用`play.py --run --seed 209`查看代表性失败。教师逐步诊断复现218决策、5075分、hero_dead，204步结束在边缘附近。详见课程093及experiments/2026-09-11-chromium-failure-review。下一步094只改变一个因素开展训练改进。


## 最新入口：094训练预算对照

运行`train.py --run --max-updates 4000`，其余配置不变；预计4255决策、4000次更新。训练完成后复制输出的评测命令（包含新权重路径）。本次实作是完整实验，不重复编写训练循环。冻结方案与待填结果见experiments/2026-09-11-chromium-training-budget/README.md；新训练尚待学习者运行。


## 最新入口：110维道具观察已集成

按学习者要求直接实现，当前训练任务为task-v2-powerups：原62维+4个12维道具槽。奖励保持原样；无需补TODO。训练用`train.py --run --max-updates 4000`，确认输出任务版本和维数后运行，随后复制输出的评测命令。旧权重仍按保存的task-v1用62维回放，默认play/evaluate路径依然是旧基线，测试新模型必须指定checkpoint。

已验证原生接口、观察契约、一次真实参数更新及新旧模型加载运行；尚无新增观察的正式4000更新结果。原生要求hello.powerups能力，不应对缺少道具字段的旧二进制静默当作空道具。


训练预算现已简化：只设置`--max-updates`，不再设置总决策上限。例：`train.py --run --max-updates 40000`。决策数仍记录；每局250决策后重开，累计训练一直进行到指定更新次数。旧实验记录中的5000总决策上限是历史配置。


## 当前训练入口：无窗口与N环境并行

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 40000 --num-envs 8
```

默认num_envs=1，训练均使用真正无显示服务模式；无需Xvfb，不创建窗口或GL上下文。N个游戏共享一个模型和经验池，max_updates是全体累计更新次数。每局250决策后单独重开，最终预算精确停止。评测也默认headless；GUI回放仍用play.py及明确checkpoint路径。

预分配TensorReplay减少反复构建批量张量，环形覆盖避免移动整个库存；保留原Replay供教学与对照。记录每步经验但每1000决策才刷新日志，异常结束时可能失去最后一小段未刷新的日志。并行分批选动作、按环境顺序收集，训练结束/异常后关闭所有游戏。不得同时调用同一个Runtime的多个请求。

4000更新对照：旧单环境9.90秒，新1/4/8环境中位数2.34/2.21/2.13秒（新方案各3次）。主要收益来自等待/数据转换优化，并行额外提升有限；不代表8倍加速。单环境优化前后权重相同；N不同会改变采样与更新时序，不能把不同N策略差异只解释成运行速度。详细记录见experiments/2026-09-11-chromium-training-throughput。


## 当前任务：task-v3-events

默认训练使用110维观察与事件奖励：击毁+1、拾取+0.2、实际损命-5、过关+20。漏机已经扣命，不重复罚；漏接原游戏分数不作为训练奖励。`--task-version task-v2-powerups`可重建旧奖励基线，旧v1/v2权重仍按自身版本回放。Runtime要求episode_events能力，缺失会报错。

训练steps日志新增本步events，评测新增每局events与mean_events（击毁、漏机、损命、拾取、漏接及两类得分）。40k新旧对照已跑完，均20局死亡，不能宣称已解决问题；详见experiments/2026-09-11-chromium-event-reward。无需立即再跑40万更新。


## 可选84维观察实验

默认仍是110维task-v3-events。加上明确版本参数试用压缩版：

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 40000 --num-envs 8 --task-version task-v4-no-motion
```

v4删除玩家keyboard_motion两项、8颗敌弹的16项速度、4个道具的8项预计位移。
保留位置、资源、对象标志、道具补给系数和类型，奖励与v3完全一致。
敌机原本没有速度输入。网络、经验池和权重元数据按版本选维数；
旧模型仍能回放，但110维权重不能直接作为84维网络的初始权重。
运行终端打印的评测/回放命令选择本次权重；默认回放路径不会自动跟随最新训练。

单次40k试验击毁改善、拾取下降，耗时基本相同，尚不能证明稳定收敛。
记录：[[experiments/2026-09-11-chromium-compact-observation/README]]。


## 当前训练默认：task-v5-shield-damage

本节覆盖上文历史默认：训练现默认v5，84维；奖励在v4基础上减去
本步实际受伤掉盾量/100。例：受损40扣0.4，受损40并损命1共扣5.4。
自然衰减、补给、重生和结束时清零不计作受伤；事件包含碰撞等受伤，非仅子弹。
需新版原生shield_damage能力，不能用缺失字段伪造零值。

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 400000 --num-envs 32
```

原命令若带`--task-version task-v4-no-motion`，会继续使用旧奖励；要用新奖励，
删除该参数或改为`--task-version task-v5-shield-damage`。
旧权重回放仍使用权重内保存的版本，不会自动获得新版训练效果。

进度条按总更新数显示，终端每0.5秒刷新，重定向每5秒输出，结束强制显示100%。
记录：[[experiments/2026-09-11-chromium-shield-reward/README]]。


## CPU可选加速更新

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 400000 --num-envs 32 --update-backend scripted
```

固定普通SGD更新编译后执行，32环境三组10000更新对照吞吐约提升18%，
逐步日志与最终参数完全一致。经验池共享内存写入优化已默认启用。
由于当前Python3.14上PyTorch对torch.jit.script有兼容性警告，编译入口保持可选；
默认eager。遇到编译兼容问题可用`--update-backend eager`运行学习版。
推理权重保持原格式。不能用该结果声称更快收敛。
当前.venv为CPU版PyTorch，CUDA尚未实测。
记录：[[experiments/2026-09-11-chromium-replay-throughput/README]]。


## 当前默认：36维观察与近期平均奖励

训练现在默认task-v6-36：自身8项、最近2敌机×3、最近4敌弹×3、最近1道具×10。
保留当前v5奖励计算（含用户改成的击毁×10、拾取+0.2、损命-5、过关+20、受伤掉盾/100惩罚）。
只缩减槽位，不变更排序、缩放、动作和网络隐藏层。旧权重仍按其版本加载。

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 40000 --num-envs 24 --task-version task-v6-36
```

不要继续带旧v4参数，否则仍训练84维且没有护盾惩罚。确认启动打印task-v6-36、36维。
新输入形状需要重新训练；已有84维权重不能直接装进36维网络。

进度条“近1000决策均奖”统计最近1000条经验的reward均值，
不足1000条按实际数量计算；包括预填经验，多个环境按经验入池顺序合并。
每条经验都计入，终端每0.5秒原行刷新；输出重定向时每5秒打印，结束强制刷新。
这是每决策奖励，不是每局总奖励、原始游戏分数或独立评测。
详见[[experiments/2026-09-11-chromium-36-observation/README]]。



## 探索率随本次总预算自动调整

前256次决策完全随机；预填结束后，在剩余决策预算前80%内将epsilon
从1.0线性降到0.05，之后保持0.05。开新局不重启，多环境累计计数。
只设置max_updates，不需要再改固定衰减步数或第二个停止上限。

当前每条可学习经验更新一次，总决策数=learning_starts-1+max_updates。
例如40000更新：总决策40255，预填256次，剩余39999次；
衰减跨度ceil(39999×0.8)=32000，约最后8000次决策保持0.05。
配置保存epsilon_start/end/decay_fraction；启动打印推导预算，进度条显示ε。
评测仍直接选最大Q动作。
记录：[[experiments/2026-09-11-chromium-epsilon-decay/README]]。


## 当前训练默认：v7逐次命中反馈

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 40000 --num-envs 24 --task-version task-v7-hit-feedback
```

36维不变。每次子弹实际削血，立即按削血/敌机初始血量给分，子弹击毁额外+2；
总伤害裁剪到剩余血量。碰撞和重生爆炸清场不再当作射击奖励。
拾取+0.2、损命-5、受伤掉盾/100惩罚、过关+20保持不变。
旧v4/v6显式参数仍选择旧奖励，注意启动版本。
详细机制与验证：[[experiments/2026-09-11-chromium-reward-redesign/README]]。

## 2026-09-11：五动作与稳定性修复

当前默认任务`task-v8-five-actions`：动作0原地、1上、2下、3左、4右，全部保持开火。没有斜向或停止开火动作。底层通过原生9..13实现，不需要修改C++。固定开火仍会消耗强化武器弹药；原地表示松开移动键，游戏已有惯性不会瞬间清零。

观察36维、奖励与v7一致；网络输出从18项改为5项。训练、评测随机基线、诊断、GUI回放都按检查点任务版本选择动作数量与映射。旧18动作检查点仍可回放；新5动作策略从头训练，不能直接使用旧输出层。

```bash
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 100000 --num-envs 24
# 使用训练结束打印的真实路径替换PATH：
.venv/bin/python exercises/chromium_dqn/evaluate.py --run --checkpoint PATH
.venv/bin/python exercises/chromium_dqn/play.py --run --checkpoint PATH --seed 1
.venv/bin/python exercises/chromium_dqn/diagnose.py --checkpoint PATH --seed 1
```

旧模型贴角落已复现为隐藏层全零、各状态Q值完全相同。训练默认损失改为Huber，每1000次更新及最后一次检查真实观察上的隐藏层与输出；失活或非有限值时保存`diagnostic.pt`、`failure.json`并停止。每10000次更新保存`policy-update-XXXXXXXXX.pt`，最终保存`policy.pt`。这些是推理权重，不包含完整断点续训状态。`health.jsonl`记录检测结果，`source_sha256.json`记录训练源码指纹。检测通过不代表学会游戏，也不能覆盖所有策略退化形式。

复现旧任务用`--task-version task-v7-hit-feedback`；旧损失对照用`--loss-kind mse`。比较5与18动作时，两组都使用Huber，其他配置相同，避免混淆动作修改与稳定性修复。

完整证据见[稳定性审计](../../experiments/2026-09-11-chromium-stability-audit/README.md)。
