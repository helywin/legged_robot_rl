#!/usr/bin/env python3
"""亲手补全一套真实 CartPole DQN 训练器。

这不是固定答案检查器。完成 TODO 后，本文件会真的执行 30,000 个环境步、
更新在线网络、冻结评估、保存 PyTorch 检查点，并导出 ONNX 推理模型。

从仓库根目录运行：

    .venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/learner_train.py

允许修改的位置：只修改标有 ``TODO 1`` ～ ``TODO 5`` 的函数体。先不要改固定
配置，否则多个因素一起变化后，就无法判断训练成败来自代码还是超参数。

你要实现的五段职责：

1. ``CartPoleQNetwork``：把 ``[batch, 4]`` 观察变成 ``[batch, 2]`` Q 值；
2. ``epsilon_at_step``：计算当前环境步的探索概率；
3. ``choose_training_action``：按 epsilon-greedy 选动作；
4. ``update_online_network``：一批经验完成一次 loss → grad → 参数更新；
5. ``run_training_loop``：连接环境交互、经验保存、抽样更新、目标同步和重置。

每个 TODO 上方都写明输入、输出、可用接口和机制顺序。未完成时脚本会给出友好
提示，不会吐出一长串无意义 traceback。

训练成功的主要现象：最终冻结评估平均回报达到 200。由于神经网络训练仍可能有
波动，偶发未过线时先保留完整输出，不要为了“刷 PASS”随意改多个参数。

成功后生成：

    artifacts/cartpole-dqn-from-scratch/online-network.pt
    artifacts/cartpole-dqn-from-scratch/online-network.onnx
    artifacts/cartpole-dqn-from-scratch/metrics.json

随后可以看 GUI：

    .venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/watch.py

并验证 ONNX 与 PyTorch 数值一致：

    .venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/verify_onnx.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import fmean
import time

import gymnasium as gym
import numpy as np
import onnx
import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    """本次受控实验的固定条件；第一次实现时不要修改。"""

    seed: int = 20260904
    total_environment_steps: int = 30_000
    replay_capacity: int = 30_000
    warmup_steps: int = 1_000
    batch_size: int = 64
    discount_factor: float = 0.99
    learning_rate: float = 0.001
    target_sync_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    evaluation_episodes: int = 20
    smoke_success_mean_return: float = 200.0


@dataclass(frozen=True)
class Transition:
    """一次环境动作留下的因果记录。"""

    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool


@dataclass(frozen=True)
class TrainingLoopResult:
    """主循环结束后交给评估和保存阶段的数据。"""

    update_count: int
    completed_episode_returns: list[float]
    progress_records: list[dict[str, float | int]]
    training_seconds: float


class ReplayBuffer:
    """保存旧经验并随机抽样；抽样不会删除经验。"""

    def __init__(self, capacity: int) -> None:
        self._transitions: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._transitions)

    def add(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def sample(
        self,
        batch_size: int,
        random_generator: random.Random,
    ) -> list[Transition]:
        return random_generator.sample(self._transitions, batch_size)


class CartPoleQNetwork(nn.Module):
    """输入四项观察，输出向左和向右两个动作的 Q 值。"""

    layers: nn.Sequential

    def __init__(self) -> None:
        super().__init__()

        # 1A：创建 4 → 64 → ReLU → 2 的网络，并赋给 self.layers。
        #
        # 只能使用下面三个已经学过的接口：
        #   nn.Linear(in_features=..., out_features=...)
        #   nn.ReLU()
        #   nn.Sequential(...)
        #
        # 输入 4：CartPole 的位置、速度、杆角度、杆角速度。
        # 隐藏 64：让网络有 64 个可组合的特征响应。
        # 输出 2：动作 0（向左）和动作 1（向右）的 Q 值。
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        """保留 Module 调用机制，同时让 VSCode 知道返回值是 Tensor。"""
        return super().__call__(observations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1B：把 observations 交给 self.layers，并返回结果。
        # 输入形状：[batch, 4]；输出形状：[batch, 2]。
        return self.layers(observations)


def set_reproducible_seed(seed: int) -> None:
    """固定本实验用到的 Python、NumPy 和 PyTorch 随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def epsilon_at_step(
    environment_step: int,
    config: TrainingConfig,
) -> float:
    """返回当前环境步的探索概率。

    2：实现线性下降。

    已知：
      - 第 0 步返回 config.epsilon_start，也就是 1.0；
      - 到 config.epsilon_decay_steps 时返回 config.epsilon_end，也就是 0.05；
      - 之后一直保持 0.05，不能继续降成负数。

    建议分两步写：
      1. progress = min(environment_step / epsilon_decay_steps, 1.0)
      2. 从 start 沿着 (end - start) 走 progress 这么远
    """
    progress = min(environment_step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def choose_training_action(
    online_network: CartPoleQNetwork,
    observation: np.ndarray,
    epsilon: float,
    random_generator: random.Random,
) -> int:
    """按 epsilon-greedy 返回动作 0 或动作 1。

    3 的机制顺序：
      1. random_generator.random() 产生 [0, 1) 的随机数；
      2. 小于 epsilon 时，用 random_generator.randrange(2) 随机探索；
      3. 否则把 observation 转成 float32 Tensor；
      4. 在 torch.no_grad() 中调用在线网络；
      5. q_values.argmax().item() 取得最大 Q 值的动作，并转成 int。

    这里不需要 batch 维：单个 [4] 输入会得到单个 [2] 输出。
    """
    if random_generator.random() < epsilon:
        action = random_generator.randrange(0, 2)
        return action
    else:
        obs = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            q_values = online_network(obs)
            action = q_values.argmax().item()
            return int(action)


def update_online_network(
    online_network: CartPoleQNetwork,
    target_network: CartPoleQNetwork,
    optimizer: torch.optim.Optimizer,
    sampled_transitions: list[Transition],
    discount_factor: float,
) -> float:
    """用一批回放经验完成一次在线网络更新，并返回 mean loss。

    4 必须保持下面的因果顺序：

      A. 用列表推导式分别取出 observation、action、reward、
         next_observation、terminated，并转成批量 Tensor。
      B. optimizer.zero_grad(set_to_none=True) 清掉上次梯度。
      C. 在线网络处理旧观察；用 gather 按每行真实 action 取 selected Q。
      D. 在 torch.no_grad() 中让目标网络处理下一观察，取每行最大 Q，
         再计算 target = reward + gamma * best_next_q * (~terminated)。
      E. 每条平方误差取 mean，得到一个标量 loss。
      F. loss.backward() 写入在线参数 .grad。
      G. optimizer.step() 读取 .grad，修改在线参数。
      H. 返回 mean_loss.item()。

    可直接使用的 Tensor 写法：

      torch.tensor([...], dtype=torch.float32)
      torch.tensor([...], dtype=torch.long)
      torch.tensor([...], dtype=torch.bool)
      actions.unsqueeze(dim=1)
      values.gather(dim=1, index=...).squeeze(dim=1)
      values.max(dim=1).values
      (~terminated).to(dtype=rewards.dtype)

    形状目标：
      observations       [batch, 4]
      actions            [batch]
      all_action_q       [batch, 2]
      selected_q         [batch]
      target_q           [batch]
      mean_loss          []，即单个标量
    """
    observations = torch.tensor([
            transition.observation for transition in sampled_transitions
        ], dtype=torch.float32)

    actions = torch.tensor([
            transition.action for transition in sampled_transitions
        ], dtype=torch.long)
    
    rewards = torch.tensor([
            transition.reward for transition in sampled_transitions
        ], dtype=torch.float32)
    
    next_observations = torch.tensor([
            transition.next_observation for transition in sampled_transitions
        ], dtype=torch.float32)

    terminateds = torch.tensor([
            transition.terminated for transition in sampled_transitions
        ], dtype=torch.bool)


    optimizer.zero_grad(True)

    all_action_q_values = online_network(observations)
    selected_q_values = all_action_q_values.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)
    with torch.no_grad():
        next_q_values = target_network(next_observations)
        best_next_q_value = next_q_values.max(dim=1).values
        future_mask = (~terminateds).to(dtype=torch.float32)
        target_q_value = rewards + best_next_q_value * discount_factor * future_mask

    per_item_losses = (selected_q_values - target_q_value) ** 2
    mean_loss = per_item_losses.mean()
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()


def evaluate_frozen_policy(
    network: CartPoleQNetwork,
    base_seed: int,
    episode_count: int,
) -> list[float]:
    """关闭随机探索和参数更新，只测量贪心策略。"""
    environment = gym.make("CartPole-v1")
    returns: list[float] = []
    was_training = network.training
    network.eval()

    try:
        for episode_index in range(episode_count):
            observation, _info = environment.reset(
                seed=base_seed + episode_index
            )
            episode_return = 0.0

            while True:
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    q_values = network(observation_tensor)
                action = int(q_values.argmax().item())

                observation, reward, terminated, truncated, _info = (
                    environment.step(action)
                )
                episode_return += float(reward)
                if terminated or truncated:
                    break

            returns.append(episode_return)
    finally:
        environment.close()
        network.train(was_training)

    return returns


def run_training_loop(
    config: TrainingConfig,
    online_network: CartPoleQNetwork,
    target_network: CartPoleQNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    random_generator: random.Random,
) -> TrainingLoopResult:
    """连接训练的五种状态变化，直到环境步预算用完。

    5：这是完整训练的主时间线。你需要自己写 ``for environment_step``
    循环，但不需要处理保存、ONNX 或 GUI。

    循环前初始化：
      - environment = gym.make("CartPole-v1")
      - observation, _info = environment.reset(seed=config.seed)
      - episode_return = 0.0
      - completed_episode_returns = []
      - recent_losses = deque(maxlen=100)
      - progress_records = []
      - update_count = 0
      - started_at = time.perf_counter()

    每个 environment_step 按此顺序写：
      1. epsilon_at_step(...) 计算探索率；
      2. choose_training_action(...) 选动作；
      3. environment.step(action) 得到下一观察和反馈；
      4. 构造 Transition 并 replay_buffer.add(...)；
      5. episode_return 加 reward，observation 指向 next_observation；
      6. 若 terminated or truncated：记录回报、清零并 reset；
      7. 若 len(replay_buffer) >= warmup_steps：随机抽 batch，调用
         update_online_network(...)，记录 loss，update_count += 1；
      8. 若 update_count 是 target_sync_interval 的倍数：
         target_network.load_state_dict(online_network.state_dict())；
      9. 每 5000 环境步打印和保存一条进度。

    必须用 try/finally 保证 environment.close()。循环后返回：

      TrainingLoopResult(
          update_count=update_count,
          completed_episode_returns=completed_episode_returns,
          progress_records=progress_records,
          training_seconds=time.perf_counter() - started_at,
      )

    进度记录字典的固定键和值：

      {
          "environment_step": environment_step,
          "update_count": update_count,
          "episode_count": len(completed_episode_returns),
          "epsilon": epsilon,
          "recent_20_episode_mean_return": fmean(
              completed_episode_returns[-20:]
          ),
          "recent_100_update_mean_loss": fmean(recent_losses),
      }

    打印格式可以照自己的习惯写；信息完整即可。
    """

    environment = gym.make("CartPole-v1")
    observation, _info = environment.reset(seed=config.seed)
    episode_return = 0.0
    completed_episode_returns = []
    recent_losses = deque(maxlen=100)
    progress_records = []
    update_count = 0
    started_at = time.perf_counter()

    try:
        for env_step in range(1, config.total_environment_steps+1):
            epsilon = epsilon_at_step(env_step, config)
            action = choose_training_action(online_network, observation, epsilon, random_generator)
            next_observation, reward, terminated, truncated, _info = environment.step(action)

            replay_buffer.add(Transition(
                observation,
                action,
                float(reward),
                next_observation,
                terminated
            ))

            observation = next_observation
            episode_return += float(reward)

            if terminated or truncated:
                completed_episode_returns.append(episode_return)
                episode_return = 0.0
                observation, _info = environment.reset()

            if len(replay_buffer) >= config.warmup_steps:
                sampled_transitions = replay_buffer.sample(
                    config.batch_size,
                    random_generator
                )
                loss = update_online_network(
                    online_network,
                    target_network,
                    optimizer,
                    sampled_transitions,
                    config.discount_factor
                    )
                recent_losses.append(loss)
                update_count += 1
                if update_count % config.target_sync_interval == 0:
                    target_network.load_state_dict(online_network.state_dict())

            if env_step % 5_000 == 0:
                            recent_returns = completed_episode_returns[-20:]
                            recent_mean_return = fmean(recent_returns)
                            recent_mean_loss = fmean(recent_losses)
                            record: dict[str, float | int] = {
                                "env_step": env_step,
                                "update_count": update_count,
                                "episode_count": len(completed_episode_returns),
                                "epsilon": epsilon,
                                "recent_20_episode_mean_return": recent_mean_return,
                                "recent_100_update_mean_loss": recent_mean_loss,
                            }
                            progress_records.append(record)
                            print(
                                f"env_step={env_step:5d} "
                                f"update_count={update_count:5d} "
                                f"episodes={len(completed_episode_returns):3d} "
                                f"epsilon={epsilon:.3f} "
                                f"recent_return={recent_mean_return:7.2f} "
                                f"recent_loss={recent_mean_loss:8.4f}"
                            )

    finally: 
        environment.close()

    return TrainingLoopResult(
              update_count=update_count,
              completed_episode_returns=completed_episode_returns,
              progress_records=progress_records,
              training_seconds=time.perf_counter() - started_at,
          )

def export_network_to_onnx(
    network: CartPoleQNetwork,
    onnx_path: Path,
) -> None:
    """把训练后的纯前向 Q 网络导出为动态 batch ONNX。"""
    was_training = network.training
    network.eval()
    example_observation = torch.zeros((1, 4), dtype=torch.float32)

    try:
        torch.onnx.export(
            network,
            (example_observation,),
            str(onnx_path),
            input_names=["observation"],
            output_names=["q_values"],
            dynamic_shapes=({0: "batch_size"},),
            dynamo=True,
        )
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
    finally:
        network.train(was_training)


def train_and_save(config: TrainingConfig) -> dict[str, object]:
    """搭好固定基础设施，调用学习者写的核心，再评估和保存。"""
    set_reproducible_seed(config.seed)
    random_generator = random.Random(config.seed)

    online_network = CartPoleQNetwork()
    target_network = CartPoleQNetwork()
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=config.learning_rate,
    )
    replay_buffer = ReplayBuffer(capacity=config.replay_capacity)

    evaluation_seed = config.seed + 10_000
    untrained_returns = evaluate_frozen_policy(
        online_network,
        base_seed=evaluation_seed,
        episode_count=config.evaluation_episodes,
    )

    loop_result = run_training_loop(
        config,
        online_network,
        target_network,
        optimizer,
        replay_buffer,
        random_generator,
    )

    trained_returns = evaluate_frozen_policy(
        online_network,
        base_seed=evaluation_seed,
        episode_count=config.evaluation_episodes,
    )
    untrained_mean = fmean(untrained_returns)
    trained_mean = fmean(trained_returns)
    smoke_passed = trained_mean >= config.smoke_success_mean_return

    artifact_directory = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "cartpole-dqn-from-scratch"
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_directory / "online-network.pt"
    onnx_path = artifact_directory / "online-network.onnx"
    metrics_path = artifact_directory / "metrics.json"

    torch.save(
        {
            "online_network_state_dict": online_network.state_dict(),
            "config": asdict(config),
        },
        checkpoint_path,
    )
    export_network_to_onnx(online_network, onnx_path)

    metrics: dict[str, object] = {
        "evidence_level": "学习者实现的 CartPole 冒烟训练与冻结评估",
        "config": asdict(config),
        "python_seed": config.seed,
        "torch_version": torch.__version__,
        "gymnasium_version": gym.__version__,
        "training_seconds": loop_result.training_seconds,
        "update_count": loop_result.update_count,
        "completed_episode_count": len(
            loop_result.completed_episode_returns
        ),
        "progress": loop_result.progress_records,
        "untrained_evaluation_returns": untrained_returns,
        "untrained_evaluation_mean_return": untrained_mean,
        "trained_evaluation_returns": trained_returns,
        "trained_evaluation_mean_return": trained_mean,
        "smoke_success_threshold": config.smoke_success_mean_return,
        "smoke_passed": smoke_passed,
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(onnx_path),
        "onnx_input": "observation: float32[batch_size, 4]",
        "onnx_output": "q_values: float32[batch_size, 2]",
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n冻结评估（相同的 20 个独立种子，无探索、无参数更新）")
    print(f"训练前平均回报：{untrained_mean:.2f}")
    print(f"训练后平均回报：{trained_mean:.2f}")
    print("随机策略历史基线：21.40")
    print(
        f"冒烟通过线：{config.smoke_success_mean_return:.2f} "
        f"{'PASS' if smoke_passed else 'FAIL'}"
    )
    print(f"训练耗时：{loop_result.training_seconds:.2f} 秒")
    print(f"PyTorch 检查点：{checkpoint_path}")
    print(f"ONNX 推理模型：{onnx_path}")
    print("ONNX 契约：observation[batch,4] → q_values[batch,2]")
    print(f"指标：{metrics_path}")
    return metrics


def main() -> None:
    config = TrainingConfig()
    print("亲手实现 CartPole DQN")
    print(
        f"单环境，environment_steps={config.total_environment_steps}，"
        f"warmup={config.warmup_steps}，batch={config.batch_size}，"
        f"seed={config.seed}"
    )
    try:
        metrics = train_and_save(config)
    except NotImplementedError as error:
        print(f"\n训练器尚未完成：{error}")
        print("请从 TODO 1 开始，按编号逐段实现。")
        return

    if not bool(metrics["smoke_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
