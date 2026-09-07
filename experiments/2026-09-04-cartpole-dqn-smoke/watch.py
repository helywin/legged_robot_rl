#!/usr/bin/env python3
"""用 Gymnasium 原生窗口回放并诊断学习者训练好的贪心策略。

先完成并运行 learner_train.py，再从仓库根目录执行：

    .venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/watch.py

窗口中的策略没有随机探索，也不会更新网络。下方诊断区实时显示：

- 网络在动作前读到的四项观察；
- 两个动作的 Q 值和实际选择；
- 动作后环境的新状态；
- 距离轨道/杆角终止边界还有多少余量；
- 回合最终因为什么结束。

空格暂停/继续，暂停时按右箭头单步，Esc 或关闭窗口退出。
"""

from __future__ import annotations

from pathlib import Path
import argparse
from statistics import fmean

import gymnasium as gym
import numpy as np
import pygame
import torch

from learner_train import CartPoleQNetwork


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "cartpole-dqn-from-scratch"
    / "online-network.pt"
)
POSITION_LIMIT_METERS = 2.4
POLE_ANGLE_LIMIT_RADIANS = 12.0 * np.pi / 180.0


def describe_end_reason(
    observation: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> str:
    """把 Gymnasium 的布尔结束信号翻译成可见的物理原因。"""
    if truncated:
        return "TIME LIMIT: reached 500 steps"
    if not terminated:
        return "RUNNING"

    reasons: list[str] = []
    if abs(float(observation[0])) > POSITION_LIMIT_METERS:
        reasons.append("TRACK LIMIT")
    if abs(float(observation[2])) > POLE_ANGLE_LIMIT_RADIANS:
        reasons.append("POLE ANGLE LIMIT")
    return " + ".join(reasons) if reasons else "TERMINATED"


def draw_diagnostic_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    step_count: int,
    observation_before_action: np.ndarray,
    q_values: torch.Tensor,
    action: int,
    observation_after_action: np.ndarray,
    end_reason: str,
) -> None:
    """把一次“观察 → Q值 → 动作 → 下一观察”画在环境窗口上。"""
    x_before, x_dot_before, theta_before, theta_dot_before = (
        float(value) for value in observation_before_action
    )
    x_after, x_dot_after, theta_after, theta_dot_after = (
        float(value) for value in observation_after_action
    )
    q_left, q_right = (float(value) for value in q_values.tolist())
    position_margin = POSITION_LIMIT_METERS - abs(x_after)
    angle_margin_degrees = np.degrees(
        POLE_ANGLE_LIMIT_RADIANS - abs(theta_after)
    )
    action_text = "LEFT  -10 N" if action == 0 else "RIGHT +10 N"

    lines = [
        f"step {step_count:3d} | action {action}: {action_text}",
        f"Q(left)={q_left:8.3f}   Q(right)={q_right:8.3f}",
        (
            "before | "
            f"x={x_before: 6.3f}  x_dot={x_dot_before: 6.3f}  "
            f"theta={np.degrees(theta_before): 6.2f} deg  "
            f"theta_dot={theta_dot_before: 6.3f}"
        ),
        (
            "after  | "
            f"x={x_after: 6.3f}  x_dot={x_dot_after: 6.3f}  "
            f"theta={np.degrees(theta_after): 6.2f} deg  "
            f"theta_dot={theta_dot_after: 6.3f}"
        ),
        (
            f"margin | track={position_margin: 6.3f} m  "
            f"pole={angle_margin_degrees: 6.2f} deg  | {end_reason}"
        ),
    ]

    overlay = pygame.Surface((screen.get_width(), 160), pygame.SRCALPHA)
    overlay.fill((12, 18, 28, 220))
    status_color = (
        (255, 105, 105) if end_reason != "RUNNING" else (180, 225, 255)
    )
    for line_index, line in enumerate(lines):
        color = status_color if line_index == len(lines) - 1 else (235, 240, 248)
        rendered_line = font.render(line, True, color)
        overlay.blit(rendered_line, (12, 8 + line_index * 24))

    overlay.blit(font.render("SPACE pause/resume | RIGHT single step when paused | ESC quit", True, (180, 225, 255)), (12, 132))
    screen.blit(overlay, (0, screen.get_height() - 160))


def load_trained_network(checkpoint_path: Path = CHECKPOINT_PATH) -> CartPoleQNetwork:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "还没有学习者检查点。请先完成并运行 learner_train.py：\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    network = CartPoleQNetwork()
    network.load_state_dict(checkpoint["online_network_state_dict"])
    network.eval()
    return network


def watch_greedy_policy(network: CartPoleQNetwork) -> list[float]:
    # 环境只生成离屏图像；合成场景和文字后，由本窗口统一刷新一次。
    environment = gym.make("CartPole-v1", render_mode="rgb_array")
    pygame.init()
    screen = pygame.display.set_mode((1024, 560))
    pygame.display.set_caption("CartPole DQN | observation -> action -> next observation")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()
    paused = False
    returns: list[float] = []

    try:
        for episode_index in range(3):
            observation, _info = environment.reset(
                seed=20280904 + episode_index
            )
            episode_return = 0.0
            step_count = 0

            while True:
                single_step = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return returns
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return returns
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        if event.key == pygame.K_RIGHT:
                            single_step = True
                if paused and not single_step:
                    clock.tick(50)
                    continue
                observation_before_action = observation.copy()
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
                step_count += 1
                end_reason = describe_end_reason(
                    observation,
                    bool(terminated),
                    bool(truncated),
                )
                frame = environment.render()
                screen.fill((240, 240, 240))
                scene = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                screen.blit(scene, ((screen.get_width() - scene.get_width()) // 2, 0))
                draw_diagnostic_overlay(
                    screen,
                    font,
                    step_count,
                    observation_before_action,
                    q_values,
                    action,
                    observation,
                    end_reason,
                )
                pygame.display.flip()
                clock.tick(50)
                if terminated or truncated:
                    deadline = pygame.time.get_ticks() + 1_000
                    while pygame.time.get_ticks() < deadline:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT or (
                                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                            ):
                                return returns
                        clock.tick(50)
                    break

            returns.append(episode_return)
            print(
                f"GUI 回放第 {episode_index + 1} 局："
                f"{episode_return:.0f} 步，结束原因：{end_reason}"
            )
    finally:
        environment.close()
        pygame.quit()

    return returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', nargs='?', type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()
    try:
        network = load_trained_network(args.checkpoint)
    except (FileNotFoundError, NotImplementedError) as error:
        print(error)
        return

    returns = watch_greedy_policy(network)
    if returns:
        print(f"已完成 {len(returns)} 局平均：{fmean(returns):.2f} 步")
    print("注意：肉眼看起来稳定只是 GUI 回放，不代替固定种子冻结评估。")


if __name__ == "__main__":
    main()
