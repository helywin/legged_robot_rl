#!/usr/bin/env python3
"""用 Gymnasium 原生窗口回放学习者训练好的贪心策略。

先完成并运行 learner_train.py，再从仓库根目录执行：

    .venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/watch.py

窗口中的策略没有随机探索，也不会更新网络。按 Ctrl+C 可提前结束。
"""

from __future__ import annotations

from pathlib import Path
from statistics import fmean

import gymnasium as gym
import torch

from learner_train import CartPoleQNetwork


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "cartpole-dqn-from-scratch"
    / "online-network.pt"
)


def load_trained_network() -> CartPoleQNetwork:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "还没有学习者检查点。请先完成并运行 learner_train.py：\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    network = CartPoleQNetwork()
    network.load_state_dict(checkpoint["online_network_state_dict"])
    network.eval()
    return network


def watch_greedy_policy(network: CartPoleQNetwork) -> list[float]:
    environment = gym.make("CartPole-v1", render_mode="human")
    returns: list[float] = []

    try:
        for episode_index in range(3):
            observation, _info = environment.reset(
                seed=20280904 + episode_index
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
            print(f"GUI 回放第 {episode_index + 1} 局：{episode_return:.0f} 步")
    finally:
        environment.close()

    return returns


def main() -> None:
    try:
        network = load_trained_network()
    except (FileNotFoundError, NotImplementedError) as error:
        print(error)
        return

    returns = watch_greedy_policy(network)
    print(f"3 局平均：{fmean(returns):.2f} 步")
    print("注意：肉眼看起来稳定只是 GUI 回放，不代替固定种子冻结评估。")


if __name__ == "__main__":
    main()
