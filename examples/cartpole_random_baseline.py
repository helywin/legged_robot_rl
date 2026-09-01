#!/usr/bin/env python3
"""观察 CartPole 标准接口，并测量不学习的随机策略基线。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean

import gymnasium as gym


DEFAULT_EPISODES = 100
DEFAULT_SEED = 20260901


@dataclass(frozen=True)
class EpisodeResult:
    """一轮随机交互结束后的可观察结果。"""

    total_reward: float
    steps: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class BaselineSummary:
    """多轮随机基线的汇总，不包含任何训练。"""

    episodes: int
    mean_return: float
    shortest_episode: int
    longest_episode: int
    terminated_episodes: int
    truncated_episodes: int


def run_random_episode(env: gym.Env, seed: int) -> EpisodeResult:
    """用固定种子的随机动作完成一轮 CartPole。"""
    observation, info = env.reset(seed=seed)
    del observation, info
    env.action_space.seed(seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        del observation, info
        total_reward += float(reward)
        steps += 1

    return EpisodeResult(
        total_reward=total_reward,
        steps=steps,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def run_random_baseline(
    episodes: int = DEFAULT_EPISODES,
    base_seed: int = DEFAULT_SEED,
) -> BaselineSummary:
    """运行多轮可重复的随机基线并返回统计结果。"""
    if episodes < 1:
        raise ValueError("episodes 必须至少为 1")

    env = gym.make("CartPole-v1")
    try:
        results = [
            run_random_episode(env, seed=base_seed + episode)
            for episode in range(episodes)
        ]
    finally:
        env.close()

    return BaselineSummary(
        episodes=episodes,
        mean_return=mean(result.total_reward for result in results),
        shortest_episode=min(result.steps for result in results),
        longest_episode=max(result.steps for result in results),
        terminated_episodes=sum(result.terminated for result in results),
        truncated_episodes=sum(result.truncated for result in results),
    )


def print_one_step(seed: int = DEFAULT_SEED) -> None:
    """打印一次 reset 和 step，让标准接口的字段直接可见。"""
    env = gym.make("CartPole-v1")
    try:
        observation, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        action = env.action_space.sample()
        new_observation, reward, terminated, truncated, new_info = env.step(action)
    finally:
        env.close()

    print("一次标准环境交互：")
    print(f"reset_observation={[round(float(value), 6) for value in observation]}")
    print(f"reset_info={info}")
    print(f"action={action} ({'向左推' if action == 0 else '向右推'})")
    print(
        "step_result="
        f"observation={[round(float(value), 6) for value in new_observation]}, "
        f"reward={float(reward):.1f}, "
        f"terminated={bool(terminated)}, truncated={bool(truncated)}, "
        f"info={new_info}"
    )


def run_demo(episodes: int = DEFAULT_EPISODES, seed: int = DEFAULT_SEED) -> None:
    """先显示接口，再显示随机策略的基线统计。"""
    print_one_step(seed)
    summary = run_random_baseline(episodes=episodes, base_seed=seed)

    print()
    print("随机策略基线（没有 Q 值、网络或参数更新）：")
    print(f"task=CartPole-v1 episodes={summary.episodes} base_seed={seed}")
    print(f"mean_return={summary.mean_return:.2f}")
    print(
        f"shortest_episode={summary.shortest_episode} "
        f"longest_episode={summary.longest_episode}"
    )
    print(
        f"terminated_episodes={summary.terminated_episodes} "
        f"truncated_episodes={summary.truncated_episodes}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="观察 CartPole 接口并运行可重复的随机策略基线"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_demo(episodes=arguments.episodes, seed=arguments.seed)
