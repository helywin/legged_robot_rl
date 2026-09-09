"""第078课续练：让资源变化真正反映在观察里（标准库离线练习）。

任务只改一个函数：encode_resources(player)。其他连接与运行代码已完成。
输入player有damage、shields、lives_counter三个实际游戏接口同名字段。
返回新的三项float元组，顺序为：
  (damage / -500.0, shields / 500.0, lives_counter / 9.0)
第一项是当前生命条的候选比例，第二项以普通护盾为参考，第三项缩放原始备用计数。
满状态damage=-500；普通护盾参考500；备用生命在当前游戏加命/setLives路径最多9。
不要把damage当作从零累计的伤害；不取绝对值、不裁剪、不加1、不判断终止，
不修改player。shields=1000应输出2.0；lives_counter=0只表示没有备用机会，
不是当前飞机死亡；若原始计数为-1，本函数保留-1/9，终止由游戏接口另外提供。
尺度来自本轮核对的当前源码，不是所有游戏通用的常量。

已有连接
========
build_with_resources先调用078的build_observation得到运动部分，再调用你的函数，
把三项资源追加到末尾。两部分来自同一份snapshot。
默认k=2：原14项顺序保持不变，索引14/15/16分别是生命条、护盾、备用生命比例。
总长度4+5*k+3，默认17。没有子弹时资源仍然保留。
没有单独的*参数。只需要属性读取、除法和tuple返回，不需要学习新的调用语法。

先手算
======
player.damage=-480，shields=420，lives_counter=4：
  -480/-500=0.96，420/500=0.84，4/9约为0.444444。
必须根据传入player计算，不能固定返回这组数字。

运行（仓库根目录）
==================
    .venv/bin/python -m exercises.game_resource_observation

成功条件：相同运动信息但不同资源时，只有最后三项中的对应位置变化；
超级护盾保留大于1的值，零备用生命不抹去其他观察，没有子弹不抹去资源。
本题把新资源实际接入观察，不只是固定答案填空。输出仍是候选局部观察，
尚缺弹药等信息；没有训练策略、实现完整环境或验证游戏效果。
"""
from dataclasses import dataclass, replace
import math

from exercises.combined_game_observation import build_observation
from exercises.bullet_motion_observation import BulletSample


@dataclass(frozen=True)
class PlayerSample:
    position: tuple[float, float, float]
    keyboard_motion: tuple[float, float]
    damage: float
    shields: float
    lives_counter: int
    score: float = 0.0


@dataclass(frozen=True)
class SnapshotSample:
    player: PlayerSample
    enemy_bullets: tuple[BulletSample, ...]


def encode_resources(player: PlayerSample) -> tuple[float, float, float]:
    # 从player读取三个资源字段，按题目顺序缩放后返回三项tuple。
    return (player.damage / -500.0, player.shields / 500.0, player.lives_counter / 9.0)


def build_with_resources(snapshot: SnapshotSample, k: int = 2) -> tuple[float, ...]:
    movement_values = build_observation(snapshot, k=k)
    resource_values = encode_resources(snapshot.player)
    return movement_values + resource_values


def check(label, snapshot, expected_resources, expected_movement, k=2):
    before = repr(snapshot)
    obs = build_with_resources(snapshot, k=k)
    assert repr(snapshot) == before, f"{label}：不能修改输入"
    assert isinstance(obs, tuple) and len(obs) == 7+5*k, f"{label}：检查返回长度与平铺结构"
    assert all(type(v) is float and math.isfinite(v) for v in obs), f"{label}：需要有限float"
    print(f"{label}: 长度={len(obs)} 资源={obs[-3:]}")
    assert obs[:-3] == expected_movement, f"{label}：资源改变不能改动运动观察"
    assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(obs[-3:],expected_resources)), f"{label}：检查字段顺序、尺度和符号"


def main():
    player = PlayerSample((2.0,1.0,0.0), (6.0,-2.0), -480.0, 420.0, 4)
    bullet = BulletSample(10, (2.0,5.0,0.0), (0.0,-1.0,0.0))
    snapshot = SnapshotSample(player, (bullet,))
    movement = build_observation(snapshot)
    try:
        check("受伤后的样本", snapshot, (0.96,0.84,4/9), movement)
        for label, changed, expected in (
            ("只改生命条", replace(player,damage=-100.0), (0.2,0.84,4/9)),
            ("只改护盾为超级护盾", replace(player,shields=1000.0), (0.96,2.0,4/9)),
            ("只改备用生命为零", replace(player,lives_counter=0), (0.96,0.84,0.0)),
            ("保留负的原始计数", replace(player,lives_counter=-1), (0.96,0.84,-1/9)),
            ("备用计数达到9", replace(player,lives_counter=9), (0.96,0.84,1.0)),
        ):
            check(label, replace(snapshot,player=changed), expected, movement)
        empty = replace(snapshot,enemy_bullets=())
        check("没有子弹仍保留资源", empty, (0.96,0.84,4/9), build_observation(empty))
        check("一颗容量对应12项", snapshot, (0.96,0.84,4/9), build_observation(snapshot,k=1), k=1)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。完整题目在本文件开头。")
        return
    print("练习通过：资源差异已进入观察；未验证策略如何利用这些信息。")


if __name__ == "__main__":
    main()
