#!/usr/bin/env python3
"""一个只依赖 Python 标准库的确定性 FrozenLake 环境。"""

from __future__ import annotations

import argparse
import random


LEFT = "left"
DOWN = "down"
RIGHT = "right"
UP = "up"
ACTIONS = (LEFT, DOWN, RIGHT, UP)
ACTION_NAMES = {
    LEFT: "向左",
    DOWN: "向下",
    RIGHT: "向右",
    UP: "向上",
}
ACTION_DELTAS = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}

DEFAULT_MAP = (
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
)


class FrozenLakeEnv:
    """4×4 冰湖：动作确定生效，终点奖励为 1，其余奖励为 0。"""

    def __init__(self, max_steps: int = 20) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须至少为 1")
        self.map = DEFAULT_MAP
        self.rows = len(self.map)
        self.cols = len(self.map[0])
        self.max_steps = max_steps
        self.start_position = self._find_tile("S")
        self.position = self.start_position
        self.step_count = 0
        self.terminated = False
        self.truncated = False

    @property
    def observation_count(self) -> int:
        return self.rows * self.cols

    @property
    def action_space(self) -> tuple[str, ...]:
        return ACTIONS

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def _find_tile(self, wanted: str) -> tuple[int, int]:
        for row, line in enumerate(self.map):
            for col, tile in enumerate(line):
                if tile == wanted:
                    return row, col
        raise ValueError(f"地图缺少 {wanted}")

    def _observation(self) -> int:
        row, col = self.position
        return row * self.cols + col

    def _tile(self) -> str:
        row, col = self.position
        return self.map[row][col]

    def _info(self) -> dict[str, object]:
        return {
            "position": self.position,
            "tile": self._tile(),
            "step_count": self.step_count,
        }

    def reset(self) -> tuple[int, dict[str, object]]:
        """重置环境并返回初始观察和辅助信息。"""
        self.position = self.start_position
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        return self._observation(), self._info()

    def step(
        self, action: str
    ) -> tuple[int, float, bool, bool, dict[str, object]]:
        """执行动作，返回观察、奖励、终止、截断和辅助信息。"""
        if self.done:
            raise RuntimeError("episode 已结束，请先调用 reset()")
        if action not in ACTION_DELTAS:
            raise ValueError(f"未知动作：{action}")

        row, col = self.position
        row_delta, col_delta = ACTION_DELTAS[action]
        next_row = min(max(row + row_delta, 0), self.rows - 1)
        next_col = min(max(col + col_delta, 0), self.cols - 1)
        self.position = (next_row, next_col)
        self.step_count += 1

        tile = self._tile()
        self.terminated = tile in ("H", "G")
        self.truncated = not self.terminated and self.step_count >= self.max_steps
        reward = 1.0 if tile == "G" else 0.0
        return (
            self._observation(),
            reward,
            self.terminated,
            self.truncated,
            self._info(),
        )

    def render(self) -> str:
        """返回适合在终端中观察的地图文本。"""
        lines = []
        for row, map_line in enumerate(self.map):
            cells = []
            for col, tile in enumerate(map_line):
                cells.append("A" if (row, col) == self.position else tile)
            lines.append("".join(f"[{cell}]" for cell in cells))
        return "\n".join(lines)


def result_name(info: dict[str, object], truncated: bool) -> str:
    if info["tile"] == "G":
        return "到达终点"
    if info["tile"] == "H":
        return "掉进冰洞"
    if truncated:
        return "步数用完"
    raise ValueError("episode 尚未结束")


def run_random_episode(seed: int, max_steps: int) -> str:
    """运行一轮尚未学习的随机策略，打印完整交互。"""
    env = FrozenLakeEnv(max_steps=max_steps)
    rng = random.Random(seed)
    observation, info = env.reset()
    print("episode 开始")
    print(env.render())
    print(f"observation={observation} position={info['position']}")

    terminated = False
    truncated = False
    while not (terminated or truncated):
        old_observation = observation
        action = rng.choice(env.action_space)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={env.step_count:2d} "
            f"observation={old_observation:2d} "
            f"action={ACTION_NAMES[action]} "
            f"new_observation={observation:2d} "
            f"reward={reward:+.1f} "
            f"terminated={terminated} "
            f"truncated={truncated}"
        )
        print(env.render())

    result = result_name(info, truncated)
    print(f"episode 结束: result={result} steps={env.step_count}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_random_episode(args.seed, args.max_steps)
