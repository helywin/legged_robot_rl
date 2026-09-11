# Chromium DQN 独立实作项目

状态：原生reset/seed已补齐并推送，新runtime已从零实现且通过真实游戏/故障检查；Task由学习者实现后按其要求由教师补齐，逻辑检查及教师运行的真实两回合已通过；DQN尚未实现。Python从零编写，不复制或导入旧课程实现。旧文件保留为历史记录。

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
