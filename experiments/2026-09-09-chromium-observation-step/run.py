"""真实游戏17项观察对照：唯一改变动作IDLE或RIGHT，固定执行5个tick。

在仓库根目录依次运行：
  .venv/bin/python experiments/2026-09-09-chromium-observation-step/run.py --action idle
  .venv/bin/python experiments/2026-09-09-chromium-observation-step/run.py --action right

每次新开1个同步GUI，先执行1个共同IDLE初始化画面，再执行5步指定动作。
不需要操作键盘，不训练网络；初始化后及每步暂停0.7秒方便观看，末尾停留3秒。
观察问题：RIGHT组飞机x与观察第0项是否一起增加？IDLE组同一阶段怎样变化？
成功条件：每步17项有限float、观察第0项等于同一快照x/10、执行步数准确，
编码不改变原始状态。方向效果需真实比较；IDLE不是冻结世界。
这不是固定seed实验，不比较跨进程全部敌机/子弹/分数是否相同。
当前尺度沿用教学配置，尚未完成最终观察设计；不代表训练或完整环境通过。
"""
import argparse
import math
from pathlib import Path
import sys
import time

# 让直接运行此文件时能够导入仓库内已完成的exercises。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chromium_rl import Action, GameClient
from exercises.game_resource_observation import build_with_resources


def show(snapshot, tick):
    observation = build_with_resources(snapshot)
    assert isinstance(observation, tuple) and len(observation) == 17
    assert all(type(v) is float and math.isfinite(v) for v in observation)
    assert math.isclose(observation[0], snapshot.player.position[0]/10, abs_tol=1e-9)
    print(f"tick={tick} world_xy={snapshot.player.position[:2]} "
          f"player_obs={observation[:4]} bullets={len(snapshot.enemy_bullets)} "
          f"resource_obs={observation[-3:]}", flush=True)
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("idle", "right"), required=True)
    args = parser.parse_args()
    action = Action.IDLE if args.action == "idle" else Action.RIGHT
    with GameClient(synchronous=True) as game:
        first = game.step(Action.IDLE)
        if first.terminated:
            raise RuntimeError("共同初始化步骤已终止，请保留输出排查")
        original = first.snapshot
        first_obs = show(original, first.episode_tick)
        assert game.snapshot() == original, "读取/编码后状态不应自行改变"
        time.sleep(0.7)
        total = 0
        last = first
        for _ in range(5):
            last = game.step(action)
            total += last.actual_ticks
            assert last.episode_tick == first.episode_tick+total
            final_obs = show(last.snapshot, last.episode_tick)
            if last.terminated:
                print("提前终止，保留结果，不当作完整5步对照：", last.snapshot.mode, flush=True)
                break
            time.sleep(0.7)
        print(f"action={args.action} actual_ticks={total} "
              f"delta_world_x={last.snapshot.player.position[0]-original.player.position[0]:.6f} "
              f"delta_obs_x={final_obs[0]-first_obs[0]:.6f}", flush=True)
        print("结束后显示3秒；按Ctrl+C也会关闭本脚本拥有的游戏进程。", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止。")
