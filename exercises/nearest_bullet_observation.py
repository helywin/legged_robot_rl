"""第076课：把数量不固定的子弹编码成固定K个观察槽（只依赖标准库）。

问题与目标
==========
给定飞机位置和任意数量的敌方子弹，按世界坐标距离选最近K颗，
每槽输出(dx / x_scale, dy / y_scale, present)，不足K颗时补(0.,0.,0.)。
dx、dy是子弹位置减飞机位置；present真子弹为1.0，空位为0.0。
最后返回平铺的tuple，长度始终是3*K。K是编码配置，训练/使用同一模型时固定；
检查不同K只是检查函数通用性，不是让一个固定输入网络临时改变输入长度。

只修改encode_bullets内的TODO，完整实现选取、排序、缩放和补位。
player_position是世界坐标(x,y,z)；bullets是BulletSample组成的tuple，
每颗提供.id和.position=(x,y,z)。字段名对应游戏接口，样本为离线教学数据。
输入保证坐标有限、ID为不同的正整数；只考虑xy，不把z或ID放入输出。
ID仅在距离平方相等时使用：较小ID排前。不得修改输入或依赖输入排列顺序。

排序务必先用未缩放的dx*dx+dy*dy，否则横纵尺度不同可能改变最近排名。
再将选中子弹的dx、dy分别除以x_scale、y_scale。不裁剪，不缩放present。
x_scale=10、y_scale=15是教学参考尺度，不是已确认的游戏边界。
使用半宽/半高表达相对位移时，跨越整个场景可能接近正负2，这不是错误。

可用的Python工具
===============
sorted(items, key=函数)：返回新列表，按函数给出的排序值升序，不修改原输入。
排序函数可用def定义，不要求lambda；若返回(a,b)，先比a，同a再比b。
items[:k]：最多取前k项；len(values)：当前长度。
values.extend((a,b,c))：向列表末尾添加三项；tuple(values)：转为元组。
可自行在TODO中写辅助函数，不能通过针对样本返回固定结果完成题目。

返回值到底装什么（根据本次实现补充）
==================================
return tuple(result)这个返回形式是正确的；需要修改的是放进result的内容。
result是一个平铺的数字列表，不是子弹对象列表，也不是世界坐标(x,y,z)列表。
K=2时六项的含义严格为：
  result[0]：最近子弹的(子弹x-飞机x)/x_scale
  result[1]：最近子弹的(子弹y-飞机y)/y_scale
  result[2]：最近槽是否有子弹，有则1.0，没有则0.0
  result[3]：第二近子弹的(子弹x-飞机x)/x_scale
  result[4]：第二近子弹的(子弹y-飞机y)/y_scale
  result[5]：第二近槽是否有子弹，有则1.0，没有则0.0
第三项是存在标记，不是position[2]（z坐标）。空槽三项都填0.0，使用float。
例如飞机(2,1,0)、子弹A(3,3,0)，A槽应是(0.1,2/15,1.0)，不是(3,3,0)。
tuple(result)仅把已有列表转成元组，不会替你做相对位置、缩放或字段转换。
返回后，check中的actual接住这个tuple，再检查长度、类型和每项数值；
现在尚未连接神经网络，后续才会把这些数字转成输入张量。

当前代码里的items每项是(bullet, distance)，因此items[i][0]是子弹对象，
items[i][1]是排序距离。取出子弹后仍需按上面规则生成三个观察数字。
排查顺序：
1. distance用于判断世界距离，使用未缩放dx、dy；缩放发生在写入观察时。
2. sorted返回一个新列表，必须接住它；单独调用不会改变items。
   例如numbers=[3,1,2]，ordered=sorted(numbers)后，numbers仍为[3,1,2]，
   ordered才是[1,2,3]。本题按(距离平方,子弹ID)排序，不能只比较距离。
3. 有子弹的槽填两个缩放相对坐标和1.0；无子弹的槽填三个0.0。
保留你已有的循环和return结构，按以上职责修正核心数据处理即可。

手算样本
========
飞机(2,1,0)，A(id=10)在(3,3,0)，B(id=20)在(-1,1,0)，C(id=30)在(2,6,0)。
相对位置A=(1,2)、B=(-3,0)、C=(0,5)，距离平方5、9、25。
K=2时选A、B，输出(0.1,2/15,1.0,-0.3,0.0,1.0)。
仅有A时第二槽补零；零子弹时全零；真子弹与飞机重合时槽为(0.,0.,1.)。

运行（仓库根目录）
==================
    .venv/bin/python -m exercises.nearest_bullet_observation

成功条件：示例正确；改变输入排列不改变观察；只改尺度不改变选中身份；
同距按ID排序；0颗/1颗/超过K颗均输出3*K项；输入不变。
程序打印对照，错误会提示对应场景，未完成时给友好提示。
本课未加入子弹速度，不是完整躲弹观察；没有原生游戏或网络训练证据。
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BulletSample:
    id: int
    position: tuple[float, float, float]


def encode_bullets(
    player_position: tuple[float, float, float],
    bullets: tuple[BulletSample, ...],
    *, k: int = 2, x_scale: float = 10.0, y_scale: float = 15.0,
) -> tuple[float, ...]:
    if type(k) is not int or k < 1:
        raise ValueError("k必须是正整数")
    if any(not math.isfinite(s) or s <= 0 for s in (x_scale, y_scale)):
        raise ValueError("两个尺度都必须是有限正数")
    # 按题目规则构造并返回新的固定长度观察。
    items: list[tuple[BulletSample, float]] = []
    for bullet in bullets:
        dx = bullet.position[0] - player_position[0]
        dy = bullet.position[1] - player_position[1]
        distance = dx**2 + dy**2
        items.append((bullet, distance))

    items = sorted(items, key=lambda item: (item[1], item[0].id))


    result = []
    for i in range(0, k):
        if i >= len(items):
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
        else:
            result.append((items[i][0].position[0] - player_position[0])/x_scale)
            result.append((items[i][0].position[1] - player_position[1])/y_scale)
            result.append(1.0)
    return tuple(result)

def check(label, player, bullets, expected, *, k=2, x_scale=10.0, y_scale=15.0):
    before = repr((player, bullets))
    actual = encode_bullets(player, bullets, k=k, x_scale=x_scale, y_scale=y_scale)
    print(f"{label}: {actual}")
    assert repr((player, bullets)) == before, f"{label}：输入被修改"
    assert isinstance(actual, tuple) and len(actual) == 3*k, f"{label}：需要3*K项tuple"
    assert all(type(v) is float and math.isfinite(v) for v in actual), f"{label}：需要有限float"
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
               for a, b in zip(actual, expected)), f"{label}：检查排序、字段、补位和尺度"


def main():
    p = (2.0, 1.0, 0.0)
    a = BulletSample(10, (3.0, 3.0, 0.0))
    b = BulletSample(20, (-1.0, 1.0, 0.0))
    c = BulletSample(30, (2.0, 6.0, 0.0))
    expected = (0.1, 2/15, 1.0, -0.3, 0.0, 1.0)
    try:
        check("三颗中选两颗", p, (c, b, a), expected)
        check("只改输入排列", p, (a, c, b), expected)
        check("只有一颗", p, (a,), (0.1, 2/15, 1.0, 0.0, 0.0, 0.0))
        check("没有子弹", p, (), (0.0,)*6)
        check("真实重合与空位", p, (BulletSample(50, p),), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        check("容量改为一槽", p, (c, b, a), (0.1, 2/15, 1.0), k=1)
        check("容量四槽补一个空位", p, (c, b, a),
              expected + (0.0, 5/15, 1.0, 0.0, 0.0, 0.0), k=4)
        # 两颗距飞机都为1，较小ID应先出现，与输入顺序无关。
        check("同距按ID排序", p,
              (BulletSample(9, (3.0, 1.0, 0.0)), BulletSample(2, (1.0, 1.0, 0.0))),
              (-0.1, 0.0, 1.0, 0.1, 0.0, 1.0))
        # 故意采用不同横纵尺度，暴露先缩放再排序的错误。
        check("只改横纵尺度", p, (c, b, a),
              (1.0, 0.02, 1.0, -3.0, 0.0, 1.0), x_scale=1.0, y_scale=100.0)
        # 同时平移飞机和子弹，不改变它们的相对关系。
        moved = tuple(BulletSample(v.id, (v.position[0]+7, v.position[1]-4, 0.0))
                      for v in (c, b, a))
        check("整体平移", (9.0, -3.0, 0.0), moved, expected)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。完整题目在本文件开头。")
        return
    print("练习通过：可变数量子弹已编码为固定槽位；尚未验证躲弹策略。")


if __name__ == "__main__":
    main()
