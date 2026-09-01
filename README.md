# legged_robot_rl

这是我的强化学习入门笔记与实验仓库。长期目标有两个：让宇树机器狗实现盲走爬楼梯，以及沿 DQN 方向训练智能体打通 `chromium-bsu`。

使用 Obsidian 时，将本仓库目录作为 Vault 打开，然后从 [[学习主页]] 开始。按知识关系复习时打开 [[图谱/概念图谱|强化学习概念图谱]]；按课程顺序学习时打开 [[教学计划]]。

当前已经完成纯 Python 走格子 Q-learning、标准环境接口和确定性 FrozenLake 表格训练，正在学习神经网络怎样用共享参数预测并更新 Q 值。两个目标先共享基础，之后分别进入游戏 DQN 和 PPO/Isaac Lab。这里首先服务于学习与分层验证，不把小环境结果写成游戏通关，也不把仿真结果直接当作真实机器狗可用的控制器。

## Python 项目环境

纯 Python 课程使用根目录的 `pyproject.toml` 管理项目元数据和依赖，并统一在 `.venv` 中运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --editable '.[dev]'
.venv/bin/python -m unittest discover -s tests -v
```

从第 040 课开始使用 PyTorch。当前只需要 CPU 计算，先完成上面的基础安装，再执行：

```bash
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch>=2.13,<3'
```

`.venv/` 已加入 `.gitignore`，不提交到仓库。后续课程、示例和测试不使用裸 `python` 或系统 Python；新增第三方依赖时先写入 `pyproject.toml`，再安装到 `.venv`。

## Isaac Lab 环境

- Distrobox：`isaac-lab3`
- 统一入口：`/home/jiang/distrobox-homes/bin/isaac-lab`
- 已有能力：Isaac Sim GUI、Go2 平地/崎岖地形训练、策略回放、TensorBoard
- 2026-08-06 检查时容器已安装但处于停止状态；使用启动器会按需进入容器

等理解最基本的强化学习概念后，再用下面两个命令确认环境和观察官方策略：

```bash
isaac-lab check
isaac-lab demo
```

不要直接运行裸 `isaacsim`。本机启动器还负责准备 Python 和 ROS 2 运行环境。

## 两个长期目标

两个目标先作为学习方向保存，完整定义见 [双目标笔记](目标/强化学习双目标.md)：

1. [宇树机器狗盲走爬楼梯](目标/四足机器人盲走楼梯.md)；
2. [DQN 打通 Chromium B.S.U.](目标/Chromium-BSU-DQN通关.md)。

现在不直接启动任何一个复杂目标。先完成环境接口、FrozenLake 和神经网络基础，再分别进入 DQN 游戏训练与四足机器人训练。

## 从零开始的学习顺序

本节只保留路线摘要；教学内容、阶段过关标准和当前进度以 [[教学计划]] 为准。

两个长期目标都不是马上执行的第一项技术任务。学习分成共享基础和两条后续分支：

1. **零基础概念**：先弄懂控制、强化学习、观察、动作和奖励分别是什么意思。
2. **纯 Python 小环境**：用走格子程序看懂一次完整交互，不接触机器人和神经网络。
3. **表格 Q-learning**：让走格子策略从尝试中学会到达目标。
4. **标准环境接口与 FrozenLake**：把同一训练循环用于一个带失败状态的小游戏。
5. **神经网络与 DQN 基础**：理解网络为什么能替代巨大 Q 表。
6. **游戏分支**：先完成 DQN 小任务，再适配和训练 Chromium B.S.U.。
7. **机器人分支**：先学习 PPO 与 Isaac Lab，再进入 Go2 平地、仿真盲走楼梯和真机安全验证。

神经网络与 DQN 基础阶段已经完成，并已用标准库串起一条回放经验的简化 DQN 更新流程。尚未完成真实神经网络的 DQN 训练、PPO、actor/critic 或 TensorBoard。

两个分支都会沿用同一个闭环，但观察与动作的具体形状不同：

```text
观察 -> 神经网络策略 -> 动作
 ^                       |
 |------ 环境与奖励 ------|
```

真实机器人部署不属于初期实验。进入该阶段前，还需要完整机器人模型、关节限位、惯量、碰撞体、执行器参数、安全限幅和急停方案。

## 仓库结构

```text
.
├── README.md             # 项目入口与学习路线
├── pyproject.toml        # Python 项目元数据和依赖入口
├── AGENTS.md             # Codex 在本仓库中的协作规则
├── .agents/skills/       # 仓库内可复用的学习与实验 Skill
├── examples/             # 可安装的纯 Python 教学包
├── exercises/            # 场景和要求自包含的 Python 编程题
├── tests/                # 纯 Python 示例的单元测试
├── 课程/                 # 带编号的课程与学习过程
├── 概念/                 # 中文命名的语义概念节点
├── 图谱/                 # Markdown 概念索引
├── 目标/                 # 两个长期目标及证据边界
└── experiments/          # 可复现实验记录；一项实验一个目录
```

训练日志、模型检查点、TensorBoard 事件文件和 Isaac Sim 缓存通常很大，不直接提交到本仓库。实验记录中写清它们的外部位置即可。

## 当前学习任务

当前已完成 [[课程/041-pytorch-module-forward|nn.Module 怎样组织参数和前向计算]]，学习者完成参数登记并在修正向量输出后通过三组标量预测。现在进入 [[课程/042-pytorch-autograd-gradient|自动求导怎样计算参数梯度]]，等待 `backward()` 练习完成；从第 039 课开始每课都必须包含学习者亲手完成的训练环节。

当前新增入口是 `examples/dqn_training_flow_demo.py`。它与前几节示例都不依赖第三方库，也不代表已经训练真实神经网络或 DQN。

本轮神经网络基础已整理为 [029 课](课程/029-neural-network-parameters.md)至 [032 课](课程/032-update-one-shared-parameter.md)，并同步到中文概念节点与概念图谱。
