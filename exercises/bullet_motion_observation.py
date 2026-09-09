"""第077课：让观察区分同位置、不同运动方向的子弹（标准库离线练习）。

场景和唯一新连接
================
076每槽只有相对位置与存在标记，分不清同位置的一颗子弹朝哪边运动。
本课每槽改为五个float：
  (dx/x_scale, dy/y_scale, vx/motion_scale, vy/motion_scale, present)
dx、dy是子弹世界位置减飞机世界位置；vx、vy直接取velocity_per_tick的前两项。
velocity_per_tick表示每一个内部tick的世界位移，不是每秒速度；不乘0.02，
不乘请求步数，不减飞机的keyboard_motion（那个量不是飞机位移）。
存在标记真子弹为1.0，空槽的五项全为0.0；不裁剪、不取绝对值、不输出z或ID。
返回平铺tuple，长度始终5*k。k与尺度是模型输入约定，使用同一模型时固定。

你要做什么
==========
只修改encode_with_motion内的TODO，构造新的输出列表，按上述五项顺序加入
selected里的每颗子弹，不足k颗补完整空槽，最后返回tuple。
已完成的排序与选取逻辑由select_nearest提供：仍按未缩放的xy距离平方、ID排序，
最多取k颗。不要改成预测距离排序；本课唯一变化是观察增加运动信息。
允许使用for、while、len、append、extend、tuple，不要求新Python语法。
bullet.position与bullet.velocity_per_tick均为(x,y,z)形式的三个float元组；
输入坐标和位移均有限、ID互异且为正；不要修改输入或其他函数。

返回值精确布局（k=2）
====================
索引0/1：最近子弹的缩放相对x/y。
索引2/3：同一颗子弹的缩放每tick位移x/y。
索引4：第一槽是否存在。
索引5/6/7/8/9：第二槽对应的五项，空槽全零。
注意present现在位于每槽第五项；不能沿用076的第三项位置。
motion_scale默认1.0是教学参考幅度，不是从游戏数据确定的正式配置。

先预测后运行
============
飞机固定在(2,1)，子弹在(2,5)，相对位置(0,4)。
两种独立场景：向下位移(0,-1)的一颗；或向上位移(0,1)的一颗。
尺度x=10、y=10、motion=1，k=1时：
  向下：(0.0,0.4,0.0,-1.0,1.0)
  向上：(0.0,0.4,0.0, 1.0,1.0)
若飞机不动、子弹存活并按位移走一步，下一相对y分别为3和5。
这里是运动算术，不根据距离下降宣称碰撞一定发生。

运行（仓库根目录）：
    .venv/bin/python -m exercises.bullet_motion_observation

成功条件：同位置不同运动只改变运动槽；只改motion_scale不改变位置/排序；
0颗/不足k颗补位正确；真实静止子弹与空槽可区分；位置与运动始终来自同一颗。
检查器还会从观察恢复一步位置变化，用于发现错用单位或把方向符号丢掉。
没有原生游戏、网络训练或躲弹评测；增加信息不等于已学会利用它。
"""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class BulletSample:
    id: int
    position: tuple[float, float, float]
    velocity_per_tick: tuple[float, float, float]


def select_nearest(player_position, bullets, k):
    """复用076已经学会的排序规则，不按预测威胁重新选择。"""
    def key(bullet):
        dx = bullet.position[0] - player_position[0]
        dy = bullet.position[1] - player_position[1]
        return (dx*dx + dy*dy, bullet.id)
    return sorted(bullets, key=key)[:k]


def encode_with_motion(
    player_position: tuple[float, float, float], bullets: tuple[BulletSample, ...],
    *, k: int = 2, x_scale: float = 10.0, y_scale: float = 10.0,
    motion_scale: float = 1.0,
) -> tuple[float, ...]:
    if type(k) is not int or k < 1:
        raise ValueError("k必须是正整数")
    if any(not math.isfinite(s) or s <= 0 for s in (x_scale, y_scale, motion_scale)):
        raise ValueError("所有尺度必须是有限正数")
    selected = select_nearest(player_position, bullets, k)
    # 将selected编码为每槽五项的观察，不足k槽补零，再返回tuple。
    result = []
    for i in range(0, k):
        if i >= len(selected):
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
        else:
            result.append((selected[i].position[0] - player_position[0]) / x_scale)
            result.append((selected[i].position[1] - player_position[1]) / y_scale)
            result.append(selected[i].velocity_per_tick[0]  / motion_scale)
            result.append(selected[i].velocity_per_tick[1]  / motion_scale)
            result.append(1.0)
    return tuple(result)


def check(label, player, bullets, expected, *, k=2, motion_scale=1.0):
    before = repr((player, bullets))
    obs = encode_with_motion(player, bullets, k=k, motion_scale=motion_scale)
    print(f"{label}: {obs}")
    assert repr((player, bullets)) == before, f"{label}：不要修改输入"
    assert isinstance(obs, tuple) and len(obs) == 5*k, f"{label}：应返回5*K项tuple"
    assert all(type(v) is float and math.isfinite(v) for v in obs), f"{label}：需要有限float"
    assert all(math.isclose(a, b, abs_tol=1e-9) for a,b in zip(obs,expected)), f"{label}：检查每槽字段与单位"
    return obs


def main():
    p = (2.0, 1.0, 0.0)
    down = BulletSample(10, (2.0, 5.0, 0.0), (0.0, -1.0, 0.0))
    up = replace(down, velocity_per_tick=(0.0, 1.0, 0.0))
    try:
        a = check("同位置向下", p, (down,), (0.0,0.4,0.0,-1.0,1.0), k=1)
        b = check("同位置向上", p, (up,), (0.0,0.4,0.0,1.0,1.0), k=1)
        print("飞机不动的一步相对y：", a[1]*10+a[3], b[1]*10+b[3])
        assert math.isclose(a[1]*10+a[3], 3.0) and math.isclose(b[1]*10+b[3], 5.0)
        check("只改运动尺度", p, (down,), (0.0,0.4,0.0,-0.5,1.0), k=1, motion_scale=2.0)
        check("没有子弹", p, (), (0.0,)*10)
        check("只有一颗补第二槽", p, (down,), (0.0,0.4,0.0,-1.0,1.0)+(0.0,)*5)
        still = BulletSample(20, p, (0.0,0.0,0.0))
        check("静止重合子弹与空位", p, (still,), (0.0,0.0,0.0,0.0,1.0)+(0.0,)*5)
        near = BulletSample(30, (3.0,1.0,0.0), (0.2,0.3,0.0))
        far = BulletSample(40, (12.0,1.0,0.0), (-5.0,0.0,0.0))
        expected = (0.1,0.0,0.2,0.3,1.0, 0.0,0.4,0.0,-1.0,1.0)
        check("位置运动成对且仍按当前距离选取", p, (far,down,near), expected)
        check("输入重新排列", p, (near,far,down), expected)
        # 相对y必须非零，才能暴露误把y_scale写成x_scale的错误。
        unequal = encode_with_motion(p, (down,), k=1, x_scale=4.0, y_scale=7.0)
        print("横纵尺度不同且相对y非零:", unequal)
        assert math.isclose(unequal[1], 4.0/7.0), "纵向相对位置必须除以y_scale"
        for scale in (0.5, 2.0):
            obs = encode_with_motion(p, (near,), k=1, x_scale=4.0, y_scale=7.0, motion_scale=scale)
            assert len(obs) == 5 and obs[4] == 1.0
            predicted_relative = (obs[0]*4 + obs[2]*scale, obs[1]*7 + obs[3]*scale)
            assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(predicted_relative,(1.2,0.3))), "检查不同尺度下一步位置恢复"
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。题目和返回值布局在本文件开头。")
        return
    print("练习通过：观察能区分位置相同、运动不同的子弹；尚未验证躲弹策略。")


if __name__ == "__main__":
    main()
