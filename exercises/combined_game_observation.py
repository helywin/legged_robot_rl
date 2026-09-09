"""第078课：从同一份快照拼接飞机与子弹观察（按学习者要求补全的示例）。

场景与核心任务
==============
075已完成飞机四项编码，077已完成每颗子弹五项编码。本课不重写它们，
build_observation已按学习者要求补全：调用两个已导入函数，传入当前snapshot及
对应配置，返回“飞机四项在前、K个子弹槽在后”的平铺float元组，长度4+5*k。
读代码时关注三步：飞机编码、子弹编码、拼接返回。
这14项只是运动与附近子弹的局部观察，没有加入damage、shields、lives_counter
或ammo_stock；不把它称为最终游戏观察。字段语义与正式选择尚需核对。

本文件SnapshotSample只保留实际快照中本课用到的字段：
  snapshot.player：075的PlayerSample，含position、keyboard_motion、score。
  snapshot.enemy_bullets：077的BulletSample元组，含id、position、velocity_per_tick。
两部分必须取自同一次调用传入的snapshot，不能缓存上一帧的位置或观察。
不得在编码函数中调用game.snapshot/step；编码不推进游戏，不重新采样。

已有函数接口（已导入，可直接调用）
================================
encode_observation(snapshot, *, position_scale=10.0, motion_scale=20.0)
  读取snapshot.player，返回飞机四项tuple；motion_scale在这里缩放控制累积量。
encode_with_motion(player_position, bullets, *, k=2, x_scale=10.0,
                   y_scale=10.0, motion_scale=1.0)
  读取飞机位置和子弹集合，返回5*k项tuple；motion_scale在这里缩放子弹每tick位移。

函数定义里单独的*是什么
======================
它不是乘法，也不是需要传入的参数；表示它后面的参数必须带名字传入。
例如旧函数要求encode_observation(snapshot, position_scale=10.0)，
而不能把尺度作为第二个无名字的位置参数传入。
本课build_observation已去掉这个限制；函数调用仍明确写参数名，方便对应物理量。
调用里的motion_scale=control_scale：左边是被调用函数的参数名，右边是当前
函数里的变量值。例如control_scale为20.0，就是把20.0交给旧函数的motion_scale。

build_observation参数如何对应
============================
position_scale：传给飞机编码的位置尺度，也传给子弹编码的x_scale和y_scale。
本课沿用075的横纵共用尺度，仅为复用现有接口，不声称场景横纵范围相同。
control_scale：只用于飞机编码的motion_scale。
bullet_motion_scale：只用于子弹编码的motion_scale。
k：传给子弹编码；改变k用于检验函数通用性，固定模型训练/使用时不能随意改变k。
注意两个旧函数都叫motion_scale，但缩放的物理量不同，不要交换或漏传参数。

Python连接工具
==============
可以先用局部变量接住两个函数返回的tuple，再拼接。
tuple的+表示按顺序拼接，例如(1.0,2.0)+(3.0,)得到(1.0,2.0,3.0)，不是逐项加法。
(first,second)则会形成嵌套元组，不符合本题输出接口。

返回布局与手算
==============
k=2时共有14项：0..3飞机自身；4..8最近子弹；9..13第二近子弹。
样本飞机position=(2,1,0)，keyboard_motion=(6,-2)，
唯一子弹position=(2,5,0)，velocity_per_tick=(0,-1,0)。默认尺度得到：
  飞机：(0.2,0.1,0.3,-0.1)
  子弹：(0.0,0.4,0.0,-1.0,1.0, 0.0,0.0,0.0,0.0,0.0)
返回就是这两段依次排列的14个float；没有子弹也必须保留飞机四项及所有空槽。

运行（仓库根目录）
==================
    .venv/bin/python -m exercises.combined_game_observation

成功条件：14项布局正确；零子弹只让子弹区归零；新快照更新相应字段；
只改control_scale或bullet_motion_scale只影响对应部分；输入不变；不同k长度正确。
本课只有离线数据连接，不创建网络、奖励或完整游戏环境。此文件现在是教师补全
示例，运行通过只说明连接检查通过，不记为学习者独立完成练习。
"""
from dataclasses import dataclass, replace
import math

from exercises.game_snapshot_observation import PlayerSample, encode_observation
from exercises.bullet_motion_observation import BulletSample, encode_with_motion


@dataclass(frozen=True)
class SnapshotSample:
    player: PlayerSample
    enemy_bullets: tuple[BulletSample, ...]


def build_observation(
    snapshot: SnapshotSample, k: int = 2,
    position_scale: float = 10.0, control_scale: float = 20.0,
    bullet_motion_scale: float = 1.0,
) -> tuple[float, ...]:
    # 第一步：从这份快照取得飞机四项；控制累积量使用control_scale。
    player_values = encode_observation(
        snapshot,
        position_scale=position_scale,
        motion_scale=control_scale,
    )

    # 第二步：同一份快照提供飞机位置和子弹；这里的运动尺度只用于子弹位移。
    bullet_values = encode_with_motion(
        snapshot.player.position,
        snapshot.enemy_bullets,
        k=k,
        x_scale=position_scale,
        y_scale=position_scale,
        motion_scale=bullet_motion_scale,
    )

    # 第三步：元组+表示拼接。前4项飞机，后5*k项子弹，不做逐项相加。
    return player_values + bullet_values

def check(label, snapshot, expected, k=2, position_scale=10.0,
          control_scale=20.0, bullet_motion_scale=1.0):
    before = repr(snapshot)
    actual = build_observation(
        snapshot,
        k=k,
        position_scale=position_scale,
        control_scale=control_scale,
        bullet_motion_scale=bullet_motion_scale,
    )
    assert repr(snapshot) == before, f"{label}：输入不能修改"
    assert isinstance(actual, tuple) and len(actual) == 4+5*k, f"{label}：需要4+5*K项平铺tuple"
    assert all(type(v) is float and math.isfinite(v) for v in actual), f"{label}：每项应为有限float"
    print(f"{label}: 长度={len(actual)} 飞机={actual[:4]} 子弹={actual[4:]}")
    assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(actual,expected)), f"{label}：检查数据来源、尺度转发和拼接顺序"


def main():
    player = PlayerSample((2.0,1.0,0.0), (6.0,-2.0))
    bullet = BulletSample(10, (2.0,5.0,0.0), (0.0,-1.0,0.0))
    first = SnapshotSample(player, (bullet,))
    hero = (0.2,0.1,0.3,-0.1)
    slot = (0.0,0.4,0.0,-1.0,1.0)
    empty = (0.0,)*5
    try:
        check("默认14项", first, hero+slot+empty)
        check("没有子弹也保留飞机", replace(first,enemy_bullets=()), hero+empty+empty)
        check("只改控制尺度", first, (0.2,0.1,0.6,-0.2)+slot+empty, control_scale=10.0)
        check("只改子弹运动尺度", first, hero+(0.0,0.4,0.0,-0.5,1.0)+empty, bullet_motion_scale=2.0)
        check("位置尺度传给两个分支", first, (0.4,0.2,0.3,-0.1)+(0.0,0.8,0.0,-1.0,1.0)+empty, position_scale=5.0)
        second = replace(first, player=replace(player, position=(3.0,2.0,0.0)))
        check("新快照的飞机位置", second, (0.3,0.2,0.3,-0.1)+(-0.1,0.3,0.0,-1.0,1.0)+empty)
        check("再读原快照不受上一调用影响", first, hero+slot+empty)
        moved = SnapshotSample(replace(player,position=(7.0,6.0,0.0)),
                               (replace(bullet,position=(7.0,10.0,0.0)),))
        check("一起平移仅自身绝对位置改变", moved, (0.7,0.6,0.3,-0.1)+slot+empty)
        check("只改分数", replace(first,player=replace(player,score=999.0)), hero+slot+empty)
        check("一槽共9项", first, hero+slot, k=1)
        check("三槽共19项", first, hero+slot+empty+empty, k=3)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。完整接口和返回值布局在文件开头。")
        return
    print("教师示例检查通过：飞机与子弹局部观察已拼接；不代表学习者独立完成或策略有效。")


if __name__ == "__main__":
    main()
