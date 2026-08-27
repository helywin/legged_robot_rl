#!/usr/bin/env python3
"""演示环境接口约定：一份通用循环代码可以运行任何遵守约定的环境。"""

from __future__ import annotations

from collections.abc import Callable

if __package__:
    from .line_world import RIGHT, LineWorldEnv
else:
    from line_world import RIGHT, LineWorldEnv


ADVANCE = "advance"
RETREAT = "retreat"
ROOM_ACTION_NAMES = {ADVANCE: "前进", RETREAT: "后退"}

Policy = Callable[[object], str]


class RoomEnv:
    """另一个遵守同样约定的环境：观察是房间名，而不是数字位置。"""

    def __init__(self, max_steps: int = 6) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须至少为 1")
        self.rooms = ("门口", "客厅", "卧室")
        self.max_steps = max_steps
        self.index = 0
        self.step_count = 0
        self.done = False

    def reset(self) -> str:
        """开始新的一轮，返回第一个观察（房间名）。"""
        self.index = 0
        self.step_count = 0
        self.done = False
        return self.rooms[self.index]

    def step(self, action: str) -> tuple[str, float, bool]:
        """执行一个动作，返回新观察、奖励和是否结束。"""
        if self.done:
            raise RuntimeError("episode 已结束，请先调用 reset()")
        if action not in (ADVANCE, RETREAT):
            raise ValueError(f"未知动作：{action}")

        self.step_count += 1
        move = 1 if action == ADVANCE else -1
        self.index = min(max(self.index + move, 0), len(self.rooms) - 1)

        reached_goal = self.index == len(self.rooms) - 1
        timed_out = self.step_count >= self.max_steps
        self.done = reached_goal or timed_out
        reward = 1.0 if reached_goal else -0.02
        return self.rooms[self.index], reward, self.done


def run_episode(env: LineWorldEnv | RoomEnv, policy: Policy) -> tuple[float, int]:
    """通用循环：只使用接口约定，不读取任何环境的内部属性。"""
    observation = env.reset()
    total_reward = 0.0
    steps = 0
    while not env.done:
        action = policy(observation)
        observation, reward, done = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward, steps


def make_fixed_policy(action: str) -> Policy:
    """返回一个永远选同一个动作的手写策略。"""

    def policy(observation: object) -> str:
        del observation
        return action

    return policy


def line_world_result(env: LineWorldEnv) -> str:
    """必须读取 LineWorldEnv 的内部字段才能判断结束原因。"""
    return "到达目标" if env.position == env.length - 1 else "步数用完"


def room_env_result(env: RoomEnv) -> str:
    """RoomEnv 的内部字段和 LineWorldEnv 不同，判断代码无法复用。"""
    return "到达目标" if env.index == len(env.rooms) - 1 else "步数用完"


def run_demo() -> None:
    """用同一个 run_episode 跑三个环境，展示接口的好处和缺口。"""
    line_env = LineWorldEnv()
    total, steps = run_episode(line_env, make_fixed_policy(RIGHT))
    print("环境1 LineWorldEnv：观察=数字位置，动作={left, right}")
    print(
        f"  总奖励={total:+.2f}  步数={steps}  "
        f"结束原因={line_world_result(line_env)}"
    )

    room_env = RoomEnv()
    total, steps = run_episode(room_env, make_fixed_policy(ADVANCE))
    print("环境2 RoomEnv：观察=房间名字符串，动作={advance, retreat}")
    print(
        f"  总奖励={total:+.2f}  步数={steps}  "
        f"结束原因={room_env_result(room_env)}"
    )

    stuck_env = RoomEnv()
    total, steps = run_episode(stuck_env, make_fixed_policy(RETREAT))
    print("环境3 RoomEnv（一直后退，永远到不了卧室）")
    print(
        f"  总奖励={total:+.2f}  步数={steps}  "
        f"结束原因={room_env_result(stuck_env)}"
    )

    print()
    print("同一个 run_episode 跑了三个环境，一行都没改。")
    print("但结束原因必须为每个环境单独写判断，因为约定没有要求环境报告原因。")
    print("标准接口（Gymnasium）的做法：step() 直接返回 terminated 和 truncated。")


if __name__ == "__main__":
    run_demo()
