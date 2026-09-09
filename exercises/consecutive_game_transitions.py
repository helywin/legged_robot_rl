"""第084课：连续动作怎样形成首尾相接的经验（标准库离线练习）。

只补全collect_transitions(game, action_ids)中的TODO。无星号参数，无网络训练。
game是本文件的ScriptedGame：按准备好的教学状态顺序返回结果，动作只记录编号，
不模拟真实运动；这些轨迹用于验证采样器，不能据此判断哪个动作更好。

已有接口与完整任务
==================
game.snapshot()：读取当前快照，不推进；可在开始读一次并逐轮接力，
也可每轮step前读取当前状态。后者在本教学源和同步游戏中同样成立。
game.step(action_id)：执行一次教学步，返回StepResultSample，包含.snapshot与.terminated。
make_transition(before, action_id, step_result)：083已完成的函数，返回五字段经验。

遍历action_ids，按顺序调用step；用本轮before、实际动作、当前结果构造经验，
将经验append到列表。随后让before引用本轮step_result.snapshot，供下一轮使用；或者下一轮
在step前重新snapshot读取同一当前状态，不强制某一种写法。
本轮terminated=True时仍保留本轮经验，然后停止，不执行余下动作。
动作列表耗尽也停止，但不要把最后一条terminated擅自改成True。
返回Transition组成的列表，不是最后一条经验，不是快照列表。
不得修改make_transition或ScriptedGame，不改变输入动作顺序或预先一次执行全部动作。

数据为何必须接力
================
教学快照分数1000→1050→1050→1100，位置x依次0→1→2→3。
动作列表(4,0,3,9)，第三步真正结束，所以第四个动作9不应执行。
预期三条reward为0.5、0、0.5，最后一条terminated=True；
第0条next_observation等于第1条observation，依此类推。
如果一直用最初的1000分作before，会得到0.5、0.5、1.0，重复计算旧得分。
before = step_result.snapshot是让局部变量指向新快照，不是修改之前保存的快照。

Python工具
==========
for action_id in action_ids：按原顺序迭代；rows.append(row)：向列表追加一条；
if result.terminated: break：退出当前循环；return rows：返回整个列表。
变量名可自定。两个已有函数的调用含义已给出，请自己把状态接力连起来。

运行（仓库根目录）
==================
    .venv/bin/python -m exercises.consecutive_game_transitions

成功条件：前后观察逐条衔接，奖励增量不重复；终止步保留且终止后不再step；
动作列表提前耗尽不伪造结束；空列表不执行动作并返回空列表。
本题只是采集经验，不写经验回放、计算loss或调用优化器；网络参数没有更新。
"""
from dataclasses import replace

from exercises.game_step_transition import StepResultSample, Transition, make_transition
from exercises.game_resource_observation import PlayerSample, SnapshotSample


class ScriptedGame:
    """有明确末端的教学结果源，不是真实游戏物理模拟。"""
    def __init__(self):
        player = PlayerSample((0.0,-3.0,0.0),(0.0,0.0),-500.0,500.0,4,score=1000.0)
        self.frames = tuple(SnapshotSample(replace(player,position=(float(i),-3.0,0.0),score=score),())
                            for i,score in enumerate((1000.0,1050.0,1050.0,1100.0)))
        self.index = 0
        self.actions = []

    def snapshot(self):
        return self.frames[self.index]

    def step(self, action_id):
        if self.index == 3:
            raise RuntimeError("教学回合已经结束，不能再调用step")
        self.actions.append(action_id)
        self.index += 1
        return StepResultSample(self.frames[self.index],self.index == 3)


def collect_transitions(game: ScriptedGame, action_ids: tuple[int, ...]) -> list[Transition]:
    # 逐步执行、构造经验、更新before，并在真正结束后停止，返回经验列表。
    result = []
    for action_id in action_ids:
        before = game.snapshot()
        step_result = game.step(action_id)
        transition = make_transition(before, action_id, step_result)
        result.append(transition)
        if step_result.terminated:
            break
    return result


def main():
    try:
        game = ScriptedGame()
        original = repr(game.frames)
        rows = collect_transitions(game,(4,0,3,9))
        assert isinstance(rows,list) and len(rows)==3, "应返回三条经验组成的列表"
        assert all(isinstance(row,Transition) for row in rows)
        for i,row in enumerate(rows):
            print(f"第{i+1}条: x观察={row.observation[0]}→{row.next_observation[0]} "
                  f"action={row.action} reward={row.reward} terminated={row.terminated}")
        assert [row.reward for row in rows] == [0.5,0.0,0.5], "before没有逐轮更新或奖励计算不正确"
        assert [row.action for row in rows] == [4,0,3] and game.actions == [4,0,3], "动作顺序或终止后执行次数不对"
        assert [row.terminated for row in rows] == [False,False,True], "终止步必须保留"
        assert all(a.next_observation==b.observation for a,b in zip(rows,rows[1:])), "相邻经验没有首尾相接"
        assert repr(game.frames)==original, "历史快照不可修改"
        short_game = ScriptedGame()
        short = collect_transitions(short_game,(4,0))
        assert len(short)==2 and not short[-1].terminated, "动作预算耗尽不等于环境真正结束"
        assert short_game.actions==[4,0]
        print("只采两步：最后terminated仍为False；不是游戏已经结束。")
        empty_game = ScriptedGame()
        empty = collect_transitions(empty_game,())
        assert empty==[] and empty_game.actions==[], "空动作列表不应执行step"
        # 同一未终止环境继续调用时，要从当前快照接起，不能缓存最初状态。
        rest = collect_transitions(short_game,(3,9))
        assert len(rest)==1 and rest[0].terminated and rest[0].reward==0.5
        assert short[-1].next_observation==rest[0].observation, "再次调用必须从当前状态继续"
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。完整接口和返回类型在文件开头。")
        return
    print("练习通过：经验逐步衔接，终止步保留且不越过回合边界；未训练网络。")


if __name__ == "__main__":
    main()
