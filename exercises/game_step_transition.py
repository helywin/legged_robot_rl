"""第083课：把一次游戏step接成一条可保存的经验（标准库离线练习）。

只补全make_transition中的TODO；外层样本、打印和检查已提供，不使用星号参数。
输入before是动作前快照，step_result是同一游戏进程、同一回合中这次动作的结果，
step_result.snapshot是动作后快照，step_result.terminated是真正结束标记。
调用者保证两份状态属于同一次动作前后，不能跨新游戏、重启、reset计算差值。
action_id是这次实际执行的整数动作编号；不会因你返回它而再次执行游戏动作。

返回一个Transition对象，字段及来源严格为：
  observation：build_with_resources(before)生成的17项tuple；
  action：传入的action_id；
  reward：(动作后player.score - 动作前player.score) / reward_scale；
  next_observation：build_with_resources(step_result.snapshot)生成的17项tuple；
  terminated：原样读取step_result.terminated，不根据生命、分数或血量自行推断。
返回类型已提供，用Transition(observation=你的变量, action=你的变量, ...)
按字段名构造即可。这里是创建对象，不是调用网络；必须根据实际输入计算，不能
返回固定样本。可以自行命名中间变量，保留输入不变。

已有接口
========
build_with_resources(snapshot)已导入，返回17项候选观察，包含位置、控制、子弹和
三项资源；当前观察不包含score。score仍可被奖励计算读取，不必为此改观察函数。
before.player.score及step_result.snapshot.player.score均为累计游戏分数。
reward_scale默认100：100分增量对应奖励1.0，只是本课候选约定，不是最终奖励。
真正结束的最后一步也保留分数差奖励，不能因为terminated=True就清零。

同一条时间线手算
================
累计分数1000→1050→1050→1100，对应三次奖励0.5、0.0、0.5，总计1.0。
第二步没有新增得分，不能再次奖励已有1050分。最后一步真正结束也保留0.5。
这只是分数增量信号，可能来自拾取或其他加分，不表示能识别具体命中事件，
也不能断言每次得分都只由当前按键造成；游戏有之前动作留下的子弹等延迟影响。

运行（仓库根目录）
==================
    .venv/bin/python -m exercises.game_step_transition

成功条件：动作前后观察来源正确；相同观察也可有不同分数增量奖励；奖励和等于
同段总分增量/尺度；最后一步奖励保留；结束标记来自结果；输入不改变。
没有创建训练器或写回放缓冲区，不运行原生游戏；这里只生成一条经验对象。
奖励方案仍缺完整任务设计，不把观察编码、奖励计算、游戏执行和网络更新混为一谈。
"""
from dataclasses import dataclass, replace
import math

from exercises.game_resource_observation import PlayerSample, SnapshotSample, build_with_resources


@dataclass(frozen=True)
class StepResultSample:
    snapshot: SnapshotSample
    terminated: bool


@dataclass(frozen=True)
class Transition:
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool


def make_transition(before: SnapshotSample, action_id: int,
                    step_result: StepResultSample, reward_scale: float = 100.0) -> Transition:
    if not math.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("reward_scale必须是有限正数")
    # 从同一次动作前后状态生成五个字段，返回Transition对象。
    """
    observation：build_with_resources(before)生成的17项tuple；
      action：传入的action_id；
      reward：(动作后player.score - 动作前player.score) / reward_scale；
      next_observation：build_with_resources(step_result.snapshot)生成的17项tuple；
      terminated：原样读取step_result.terminated，不根据生命、分数或血量自行推断。    
    """
    observation = build_with_resources(before)
    reward = (step_result.snapshot.player.score - before.player.score) / reward_scale
    next_observation = build_with_resources(step_result.snapshot)
    return Transition(observation, action_id, reward, next_observation, step_result.terminated)


def check(before, action, result, expected_reward, scale=100.0):
    original = repr((before,result))
    transition = make_transition(before,action,result,reward_scale=scale)
    assert isinstance(transition,Transition), "请返回Transition对象，不是任意tuple"
    assert repr((before,result)) == original, "不要修改输入"
    assert transition.observation == build_with_resources(before), "observation应来自动作前"
    assert transition.next_observation == build_with_resources(result.snapshot), "next_observation应来自动作后"
    assert type(transition.action) is int and transition.action == action, "记录本次实际动作编号"
    assert type(transition.reward) is float and math.isclose(transition.reward,expected_reward,abs_tol=1e-9), "检查新增分数、尺度与终止步奖励"
    assert type(transition.terminated) is bool and transition.terminated == result.terminated, "结束标记来自step结果"
    print(f"score {before.player.score:g}→{result.snapshot.player.score:g}: "
          f"reward={transition.reward} action={transition.action} terminated={transition.terminated}")
    return transition


def main():
    p = PlayerSample((0.0,-3.0,0.0),(0.0,0.0),-500.0,500.0,4,score=1000.0)
    frames = [SnapshotSample(replace(p,score=s),()) for s in (1000.0,1050.0,1050.0,1100.0)]
    try:
        rows = []
        for i, reward in enumerate((0.5,0.0,0.5)):
            row = check(frames[i],4,StepResultSample(frames[i+1],i==2),reward)
            rows.append(row)
        assert all(row.observation == row.next_observation for row in rows), "只改分数时当前观察保持不变"
        assert math.isclose(sum(row.reward for row in rows),1.0), "奖励应对应总分增量，不重复计算累计分数"
        moved = replace(frames[1],player=replace(frames[1].player,position=(1.0,-3.0,0.0)))
        check(frames[0],4,StepResultSample(moved,False),0.5)
        check(frames[0],9,StepResultSample(frames[1],False),0.25,scale=200.0)
        # 模拟游戏结果权威地报告未结束：零备用生命本身不是结束条件。
        zero = replace(frames[1],player=replace(frames[1].player,lives_counter=0))
        check(frames[0],0,StepResultSample(zero,False),0.5)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。完整字段与构造方式在文件开头。")
        return
    print("练习通过：已生成动作前后对应的经验；未执行网络训练。")


if __name__ == "__main__":
    main()
