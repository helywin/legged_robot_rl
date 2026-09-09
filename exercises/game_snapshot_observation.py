"""第075课：把原始状态转换为顺序固定的四项观察（仅标准库）。

场景：游戏快照包含飞机位置、控制累积量和分数。本课只接通飞机自身信息，
不代表完整躲弹观察。position是世界坐标(x,y,z)，不是像素；keyboard_motion
是游戏的水平/竖直控制累积量，不是真实速度，也不是本轮动作编号。

任务：只修改encode_observation中的TODO。输入snapshot提供.player.position、
.player.keyboard_motion、.player.score；输出一个新的四浮点数tuple，顺序固定为：
(x / position_scale, y / position_scale,
 水平控制累积量 / motion_scale, 竖直控制累积量 / motion_scale)。
只读输入，不加入z或score，不取绝对值、不裁剪、不修改任何输入。
position_scale和motion_scale由调用者指定为正数，默认10和20只是教学尺度，
不是游戏边界，不保证输出在[-1,1]内；校验已提供。可用float(value)转换数字，
通过snapshot.player读取字段，通过tuple索引0、1取得两个方向。
核心转换必须使用实际传入的字段和尺度，不能针对样本返回固定答案。

运行（仓库根目录）：
    .venv/bin/python -m exercises.game_snapshot_observation

先预测：位置(2,-4,0)、控制累积量(6,-2)应输出(0.2,-0.4,0.3,-0.1)。
成功条件：不同位置/控制信息反映在对应槽位；只改分数观察不变；
同一位置不同控制累积量可区分；改变尺度只改变数值表达；输入不变。
程序会展示对照并检查这些关系。可自行新增样本用于探索，无需修改检查逻辑。
本练习没有启动原生游戏、调用网络或更新参数；通过后仍不能声称策略会躲弹。
"""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class PlayerSample:
    position: tuple[float, float, float]
    keyboard_motion: tuple[float, float]
    score: float = 0.0


@dataclass(frozen=True)
class SnapshotSample:
    player: PlayerSample


def encode_observation(
    snapshot: SnapshotSample, *, position_scale: float = 10.0,
    motion_scale: float = 20.0,
) -> tuple[float, float, float, float]:
    """消费与真实快照同名的字段；离线样本只保留本课需要的部分。"""
    if any(not math.isfinite(s) or s <= 0 for s in (position_scale, motion_scale)):
        raise ValueError("两个尺度都必须是有限正数")
    # 读取本次传入的状态，按题目顺序转换为新的四项tuple。
    # (x / position_scale, y / position_scale, 水平控制累积量 / motion_scale, 竖直控制累积量 / motion_scale)
    return (snapshot.player.position[0] / position_scale, 
            snapshot.player.position[1] / position_scale,
            snapshot.player.keyboard_motion[0] / motion_scale,
            snapshot.player.keyboard_motion[1] / motion_scale
            )


def main() -> None:
    base = SnapshotSample(PlayerSample((2.0, -4.0, 0.0), (6.0, -2.0)))
    changed_motion = SnapshotSample(replace(base.player, keyboard_motion=(-6.0, 2.0)))
    changed_score = SnapshotSample(replace(base.player, score=9000.0))
    original = repr(base)
    try:
        first = encode_observation(base)
        second = encode_observation(changed_motion)
        score_only = encode_observation(changed_score)
        scaled = encode_observation(base, position_scale=5.0, motion_scale=10.0)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。题目和运行方法在本文件开头。")
        return
    for name, obs in (("原始样本", first), ("同位置、不同控制累积量", second),
                      ("只改分数", score_only), ("尺度减半", scaled)):
        print(f"{name}: {obs}")
        assert isinstance(obs, tuple) and len(obs) == 4, "请返回四项tuple"
        assert all(type(v) is float and math.isfinite(v) for v in obs), "需要有限float"
    assert all(math.isclose(a, b) for a, b in zip(first, (0.2, -0.4, 0.3, -0.1))), "检查顺序、尺度和符号"
    assert second[:2] == first[:2] and second[2:] == tuple(-v for v in first[2:]), "控制信息应能区分"
    assert first == score_only, "未选入观察的分数不能影响输出"
    assert all(math.isclose(a, 2*b) for a, b in zip(scaled, first)), "尺度减半时输出应加倍"
    assert repr(base) == original, "输入不应被修改"
    # 多组新坐标检查关系，避免只对开头样本返回固定结果。
    for x, y, mx, my in ((-3., 7., 0., 8.), (12., -1., -4., 0.), (0., 0., 2., -9.)):
        sample = SnapshotSample(PlayerSample((x, y, 99.), (mx, my), 123.))
        obs = encode_observation(sample, position_scale=2., motion_scale=4.)
        assert len(obs) == 4
        recovered = tuple(v*s for v, s in zip(obs, (2., 2., 4., 4.)))
        assert all(math.isclose(a,b) for a,b in zip(recovered, (x,y,mx,my))), "检查新样本的字段映射"
    print("练习通过：四项观察的字段顺序、缩放和信息边界正确；尚未验证游戏策略。")


if __name__ == "__main__":
    main()
