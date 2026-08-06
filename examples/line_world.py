#!/usr/bin/env python3
"""A tiny reinforcement-learning environment using only Python's standard library."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable


LEFT = "left"
RIGHT = "right"
ACTION_NAMES = {LEFT: "向左", RIGHT: "向右"}


class LineWorldEnv:
    """A five-cell world where the goal is to reach the rightmost cell."""

    def __init__(self, length: int = 5, max_steps: int = 10) -> None:
        if length < 2:
            raise ValueError("length 必须至少为 2")
        if max_steps < 1:
            raise ValueError("max_steps 必须至少为 1")
        self.length = length
        self.max_steps = max_steps
        self.position = 0
        self.step_count = 0
        self.done = False

    def reset(self) -> int:
        """Start a new episode and return the first observation."""
        self.position = 0
        self.step_count = 0
        self.done = False
        return self.position

    def step(self, action: str) -> tuple[int, float, bool]:
        """Apply one action and return observation, reward, and done."""
        if self.done:
            raise RuntimeError("episode 已结束，请先调用 reset()")
        if action not in (LEFT, RIGHT):
            raise ValueError(f"未知动作：{action}")

        self.step_count += 1
        movement = -1 if action == LEFT else 1
        self.position = min(max(self.position + movement, 0), self.length - 1)

        reached_goal = self.position == self.length - 1
        timed_out = self.step_count >= self.max_steps
        self.done = reached_goal or timed_out
        reward = 1.0 if reached_goal else -0.01
        return self.position, reward, self.done

    def render(self) -> str:
        cells = [" " for _ in range(self.length)]
        cells[-1] = "G"
        cells[self.position] = "A" if self.position != self.length - 1 else "A/G"
        return "".join(f"[{cell}]" for cell in cells)


Policy = Callable[[int, random.Random], str]


def always_right_policy(observation: int, rng: random.Random) -> str:
    """A hand-written policy: always move right."""
    del observation, rng
    return RIGHT


def random_policy(observation: int, rng: random.Random) -> str:
    """A policy that has not learned: choose an action randomly."""
    del observation
    return rng.choice([LEFT, RIGHT])


def run_episode(policy_name: str, seed: int) -> float:
    env = LineWorldEnv()
    rng = random.Random(seed)
    policies: dict[str, Policy] = {
        "right": always_right_policy,
        "random": random_policy,
    }
    policy = policies[policy_name]

    observation = env.reset()
    total_reward = 0.0
    print(f"episode 开始: {env.render()} observation={observation}")

    while not env.done:
        old_observation = observation
        action = policy(observation, rng)
        observation, reward, done = env.step(action)
        total_reward += reward
        print(
            f"step={env.step_count:2d} "
            f"observation={old_observation} "
            f"action={ACTION_NAMES[action]} "
            f"new_observation={observation} "
            f"reward={reward:+.2f} "
            f"done={done} "
            f"{env.render()}"
        )

    result = "到达目标" if env.position == env.length - 1 else "步数用完"
    print(f"episode 结束: result={result} total_reward={total_reward:+.2f}")
    return total_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("right", "random"), default="right")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_episode(args.policy, args.seed)
