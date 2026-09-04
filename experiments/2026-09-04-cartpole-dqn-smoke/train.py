#!/usr/bin/env python3
"""第一次真实 CartPole DQN 冒烟训练。

从仓库根目录运行：

    .venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/train.py

本脚本没有需要填写的 TODO，也不要求输入一堆命令行参数。所有配置集中在
``TrainingConfig``，并按“数据采集、网络更新、目标稳定、独立评估”四种职责
固定下来。学习者本课的任务是亲自运行完整训练并阅读阶段输出。
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
import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    """本次受控实验的固定条件，不需要在命令行逐项填写。"""

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
    """一次真实环境交互产生的因果记录。"""

    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool


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
    """四项 CartPole 观察经过 64 个 ReLU 单元，输出两个动作 Q 值。"""

    layers: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=4, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
        )

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        """保留 Module 调用流程，并让 VSCode 知道返回 Tensor。"""
        return super().__call__(observations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


def set_reproducible_seed(seed: int) -> None:
    """固定本实验使用的 Python、NumPy 和 PyTorch 随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def epsilon_at_step(
    environment_step: int,
    config: TrainingConfig,
) -> float:
    """让探索率从 1.0 线性下降，之后保持 0.05。"""
    progress = min(
        environment_step / config.epsilon_decay_steps,
        1.0,
    )
    return config.epsilon_start + progress * (
        config.epsilon_end - config.epsilon_start
    )


def choose_training_action(
    online_network: CartPoleQNetwork,
    observation: np.ndarray,
    epsilon: float,
    random_generator: random.Random,
) -> int:
    """训练阶段使用 epsilon-greedy，保留探索产生的新经验。"""
    if random_generator.random() < epsilon:
        return random_generator.randrange(2)

    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
    )
    with torch.no_grad():
        q_values = online_network(observation_tensor)
    return int(q_values.argmax().item())


def update_online_network(
    online_network: CartPoleQNetwork,
    target_network: CartPoleQNetwork,
    optimizer: torch.optim.Optimizer,
    sampled_transitions: list[Transition],
    discount_factor: float,
) -> float:
    """让一批回放经验共同完成一次在线参数更新。"""
    observations = torch.tensor(
        [item.observation for item in sampled_transitions],
        dtype=torch.float32,
    )
    actions = torch.tensor(
        [item.action for item in sampled_transitions],
        dtype=torch.long,
    )
    rewards = torch.tensor(
        [item.reward for item in sampled_transitions],
        dtype=torch.float32,
    )
    next_observations = torch.tensor(
        [item.next_observation for item in sampled_transitions],
        dtype=torch.float32,
    )
    terminated = torch.tensor(
        [item.terminated for item in sampled_transitions],
        dtype=torch.bool,
    )

    optimizer.zero_grad(set_to_none=True)

    all_action_q_values = online_network(observations)
    selected_q_values = all_action_q_values.gather(
        dim=1,
        index=actions.unsqueeze(dim=1),
    ).squeeze(dim=1)

    with torch.no_grad():
        next_q_values = target_network(next_observations)
        best_next_q_values = next_q_values.max(dim=1).values
        future_mask = (~terminated).to(dtype=rewards.dtype)
        target_q_values = rewards + (
            discount_factor * best_next_q_values * future_mask
        )

    per_item_losses = (selected_q_values - target_q_values) ** 2
    mean_loss = per_item_losses.mean()
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()


def evaluate_frozen_policy(
    network: CartPoleQNetwork,
    base_seed: int,
    episode_count: int,
) -> list[float]:
    """在独立环境中关闭探索和更新，只测量贪心策略。"""
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


def train(config: TrainingConfig) -> dict[str, object]:
    """按环境时间线完成采集、回放更新、同步和回合重置。"""
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

    environment = gym.make("CartPole-v1")
    observation, _info = environment.reset(seed=config.seed)
    episode_return = 0.0
    completed_episode_returns: list[float] = []
    recent_losses: deque[float] = deque(maxlen=100)
    progress_records: list[dict[str, float | int]] = []
    update_count = 0
    started_at = time.perf_counter()

    try:
        for environment_step in range(
            1,
            config.total_environment_steps + 1,
        ):
            epsilon = epsilon_at_step(environment_step, config)
            action = choose_training_action(
                online_network,
                observation,
                epsilon,
                random_generator,
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                _info,
            ) = environment.step(action)

            replay_buffer.add(
                Transition(
                    observation=tuple(
                        float(value) for value in observation
                    ),
                    action=action,
                    reward=float(reward),
                    next_observation=tuple(
                        float(value) for value in next_observation
                    ),
                    terminated=bool(terminated),
                )
            )
            episode_return += float(reward)
            observation = next_observation

            if terminated or truncated:
                completed_episode_returns.append(episode_return)
                episode_return = 0.0
                observation, _info = environment.reset()

            if len(replay_buffer) >= config.warmup_steps:
                sampled_transitions = replay_buffer.sample(
                    config.batch_size,
                    random_generator,
                )
                loss = update_online_network(
                    online_network,
                    target_network,
                    optimizer,
                    sampled_transitions,
                    config.discount_factor,
                )
                recent_losses.append(loss)
                update_count += 1

                if update_count % config.target_sync_interval == 0:
                    target_network.load_state_dict(
                        online_network.state_dict()
                    )

            if environment_step % 5_000 == 0:
                recent_returns = completed_episode_returns[-20:]
                recent_mean_return = fmean(recent_returns)
                recent_mean_loss = fmean(recent_losses)
                record: dict[str, float | int] = {
                    "environment_step": environment_step,
                    "update_count": update_count,
                    "episode_count": len(completed_episode_returns),
                    "epsilon": epsilon,
                    "recent_20_episode_mean_return": recent_mean_return,
                    "recent_100_update_mean_loss": recent_mean_loss,
                }
                progress_records.append(record)
                print(
                    f"environment_step={environment_step:5d} "
                    f"update_count={update_count:5d} "
                    f"episodes={len(completed_episode_returns):3d} "
                    f"epsilon={epsilon:.3f} "
                    f"recent_return={recent_mean_return:7.2f} "
                    f"recent_loss={recent_mean_loss:8.4f}"
                )
    finally:
        environment.close()

    training_seconds = time.perf_counter() - started_at
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
        / "cartpole-dqn-smoke"
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_directory / "online-network.pt"
    metrics_path = artifact_directory / "metrics.json"

    torch.save(
        {
            "online_network_state_dict": online_network.state_dict(),
            "config": asdict(config),
        },
        checkpoint_path,
    )

    metrics: dict[str, object] = {
        "evidence_level": "CartPole 冒烟训练与冻结评估",
        "config": asdict(config),
        "python_seed": config.seed,
        "torch_version": torch.__version__,
        "gymnasium_version": gym.__version__,
        "training_seconds": training_seconds,
        "update_count": update_count,
        "completed_episode_count": len(completed_episode_returns),
        "progress": progress_records,
        "untrained_evaluation_returns": untrained_returns,
        "untrained_evaluation_mean_return": untrained_mean,
        "trained_evaluation_returns": trained_returns,
        "trained_evaluation_mean_return": trained_mean,
        "smoke_success_threshold": config.smoke_success_mean_return,
        "smoke_passed": smoke_passed,
        "checkpoint_path": str(checkpoint_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n冻结评估（相同的 20 个独立种子，无探索、无参数更新）")
    print(f"训练前平均回报：{untrained_mean:.2f}")
    print(f"训练后平均回报：{trained_mean:.2f}")
    print(f"随机策略历史基线：21.40")
    print(
        f"冒烟通过线：{config.smoke_success_mean_return:.2f} "
        f"{'PASS' if smoke_passed else 'FAIL'}"
    )
    print(f"训练耗时：{training_seconds:.2f} 秒")
    print(f"检查点：{checkpoint_path}")
    print(f"指标：{metrics_path}")
    print("最终验收线仍是：100 回合平均至少 475，至少 90 回合达到 500 步")
    return metrics


def main() -> None:
    config = TrainingConfig()
    print("第一次真实 CartPole DQN 冒烟训练")
    print(
        f"单环境，environment_steps={config.total_environment_steps}，"
        f"warmup={config.warmup_steps}，batch={config.batch_size}，"
        f"seed={config.seed}"
    )
    metrics = train(config)
    if not bool(metrics["smoke_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
