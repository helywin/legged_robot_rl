"""第082课：真实子弹不推进与推进1tick的对照，1个GUI、0次网络训练。

仓库根目录运行：
  .venv/bin/python experiments/2026-09-09-chromium-bullet-step/run.py

脚本保持IDLE，最多600个tick等待子弹，每步观看暂停0.02秒。
遇到子弹后先等待现实1秒并读取快照，检查状态不变；再推进恰好1tick。
按稳定ID跟踪存活子弹，不能把下一帧的“最近子弹”默认当作上一帧那一颗。
观察same_snapshot=True、同一ID的预测/实际xy，以及原观察槽的五项如何对应。
出现消失子弹时跳过该对子照，直到找到可验证的存活子弹或达到上限/终止。
成功条件：非空17项编码正确；等待不推进；同一存活ID下一位置=原位置+每tick位移。
这是已有编码的真实受控实验，唯一对照变量为是否推进1tick，不改变动作、尺度。
无公共seed，结果坐标不必与教师相同；未找到有效样本明确报告未完成，不算通过。
不是训练成绩、碰撞预测、所有子弹类型验收或完整观察设计。
"""
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chromium_rl import Action, GameClient
from exercises.game_resource_observation import build_with_resources


def verify_observation(snapshot):
    obs = build_with_resources(snapshot)
    assert len(obs) == 17 and all(type(v) is float and math.isfinite(v) for v in obs)
    px, py = snapshot.player.position[:2]
    ordered = sorted(snapshot.enemy_bullets,
                     key=lambda b: ((b.position[0]-px)**2+(b.position[1]-py)**2, b.id))[:2]
    for i in range(2):
        actual = obs[4+5*i:9+5*i]
        if i >= len(ordered):
            expected = (0.0,)*5
        else:
            b = ordered[i]
            expected = ((b.position[0]-px)/10, (b.position[1]-py)/10,
                        b.velocity_per_tick[0], b.velocity_per_tick[1], 1.0)
        assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(actual,expected))
    return ordered, obs


def main():
    with GameClient(synchronous=True) as game:
        current = game.step(Action.IDLE)
        while current.episode_tick < 600 and not current.terminated:
            before = current.snapshot
            if not before.enemy_bullets:
                current = game.step(Action.IDLE)
                time.sleep(0.02)
                continue
            ordered, obs = verify_observation(before)
            tracked = ordered[0]
            print(f"before tick={current.episode_tick} id={tracked.id} type={tracked.type} "
                  f"player_xy={before.player.position[:2]} bullet_xy={tracked.position[:2]} "
                  f"per_tick_xy={tracked.velocity_per_tick[:2]}", flush=True)
            print("before第一子弹槽=", obs[4:9], flush=True)
            time.sleep(1)
            unchanged = game.snapshot() == before
            print("等待现实1秒 same_snapshot=", unchanged, flush=True)
            assert unchanged, "同步游戏未推进时快照应不变"
            previous_tick = current.episode_tick
            current = game.step(Action.IDLE)
            assert current.actual_ticks == 1 and current.episode_tick == previous_tick+1
            after = current.snapshot
            matches = [b for b in after.enemy_bullets if b.id == tracked.id]
            if not matches:
                print("该ID已消失，不从消失推断命中；寻找另一对有效样本。", flush=True)
                continue
            predicted = tuple(p+v for p,v in zip(tracked.position,tracked.velocity_per_tick))
            actual = matches[0].position
            print("推进1tick predicted_xy=", predicted[:2], "actual_xy=", actual[:2], flush=True)
            assert all(math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-5) for a,b in zip(predicted,actual))
            after_order, _ = verify_observation(after)
            print("下一帧最近两颗ID=", [b.id for b in after_order], flush=True)
            print("接口检查通过：非空槽、等待不推进、同一子弹单步运动。窗口保留3秒。", flush=True)
            time.sleep(3)
            return
        print(f"未完成：tick={current.episode_tick} mode={current.snapshot.mode}，未找到有效存活子弹对。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止，未将中断记为通过。")
