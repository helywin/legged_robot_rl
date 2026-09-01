#!/usr/bin/env python3
"""用标准库演示 DQN 经验回放的保存、淘汰和随机抽取。"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    """一次环境交互产生的一条经验。"""

    observation: int
    action: str
    reward: float
    new_observation: int
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """保存有限条旧经验，并允许随机抽取且不删除它们。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity 必须至少为 1")
        self._transitions: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._transitions)

    def add(self, transition: Transition) -> None:
        """保存新经验；容量满时自动淘汰最旧经验。"""
        self._transitions.append(transition)

    def snapshot(self) -> tuple[Transition, ...]:
        """返回当前经验快照，便于观察但不暴露内部容器。"""
        return tuple(self._transitions)

    def sample(
        self, sample_size: int, rng: random.Random
    ) -> list[Transition]:
        """随机抽取互不重复的旧经验，不从缓冲区删除。"""
        if sample_size < 1:
            raise ValueError("sample_size 必须至少为 1")
        if sample_size > len(self._transitions):
            raise ValueError("sample_size 不能大于当前经验数量")
        return rng.sample(list(self._transitions), sample_size)


def make_transition(observation: int) -> Transition:
    """创建一条便于追踪编号的演示经验。"""
    return Transition(
        observation=observation,
        action="RIGHT",
        reward=1.0 if observation == 4 else 0.0,
        new_observation=observation + 1,
        terminated=observation == 4,
        truncated=False,
    )


def observation_ids(
    transitions: tuple[Transition, ...] | list[Transition],
) -> list[int]:
    """只提取观察编号，让终端输出更直观。"""
    return [transition.observation for transition in transitions]


def run_demo() -> None:
    """加入五条经验到容量四的缓冲区，再随机抽取三条。"""
    replay_buffer = ReplayBuffer(capacity=4)

    for observation in range(5):
        replay_buffer.add(make_transition(observation))
        print(
            f"add={observation} "
            f"buffer={observation_ids(replay_buffer.snapshot())}"
        )

    before_sample = replay_buffer.snapshot()
    sampled = replay_buffer.sample(sample_size=3, rng=random.Random(7))
    after_sample = replay_buffer.snapshot()

    print(f"sampled={observation_ids(sampled)}")
    print(f"buffer_after_sample={observation_ids(after_sample)}")
    print(f"sample_removed_data={before_sample != after_sample}")


if __name__ == "__main__":
    run_demo()
