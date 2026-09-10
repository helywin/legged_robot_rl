"""第086课：复用采样循环，返回经验和停止原因（标准库离线练习）。

只修改collect_batch(game, action_id, max_decisions)里的TODO。
084的collect_transitions已经导入，不要复制或重写那个循环。

要完成的连接
============
1. 准备一个长度为max_decisions的动作tuple，每项为action_id。
   Python写法示例：(4,)*3得到(4,4,4)，逗号表示单项tuple，乘法表示重复。
2. 将game和这个动作tuple传给collect_transitions，得到经验列表。
3. 如果最后一条经验的terminated=True，stop_reason为"terminated"；
   否则为"batch_limit"，表示这批采够了，但没有结束当前回合。
4. 返回SamplingBatch(transitions=经验列表, stop_reason=停止原因)。

max_decisions已验证为正整数，输入保证game处于可继续状态。旧采样器会至少执行
一次并保存终止步，因此最后一条可以用rows[-1]访问。不要再次调用step探测是否结束。
即使实际条数恰好等于max_decisions，只要最后一条是真正终止，原因也应为terminated。
不修改经验的terminated，不生成truncated，不重启游戏；这里是批次限制不是回合限制。
若返回terminated，调用者必须结束或重开，不能继续调用同一个已结束game。

返回值为什么是对象
==================
之前只有list，外层需要自己猜为什么结束；现在result.transitions仍是那份list，
result.stop_reason明确说明原因。停止原因是采样器元数据，不加入17项观察或奖励。

手算时间线与预期
================
沿用084教学游戏：三步后真正结束，分数1000→1050→1050→1100。
第一次预算2：两条奖励[0.5,0]，reason=batch_limit，最后terminated=False。
对同一game第二次预算2：只产生最后一条奖励[0.5]，reason=terminated。
新game预算3：恰好第三条结束，reason仍为terminated，不能写batch_limit。
新game预算5：也只产生三条，不越过真实结束。

运行（仓库根目录）
==================
    .venv/bin/python -m exercises.sampling_stop_reason

成功条件：批次间前后观察衔接；停止原因准确；终止步奖励保留；不多执行step；
所有动作按本次action_id记录。核心是把已有真实采样过程封装成可消费的结果，
不是针对固定样本填字符串。game沿用离线教学结果源，不是原生运动模拟。
当前没有网络训练；真实游戏接入会复用这个契约，但其Action枚举转换由外层处理。
"""
from dataclasses import dataclass

from exercises.consecutive_game_transitions import ScriptedGame, collect_transitions
from exercises.game_step_transition import Transition


@dataclass(frozen=True)
class SamplingBatch:
    transitions: list[Transition]
    stop_reason: str


def collect_batch(game: ScriptedGame, action_id: int, max_decisions: int) -> SamplingBatch:
    if type(max_decisions) is not int or max_decisions <= 0:
        raise ValueError("max_decisions必须是正整数")
    # 复用collect_transitions采样，读取真实末条标记，返回经验与停止原因。
    action_ids = (action_id,) * max_decisions
    batch = collect_transitions(game, action_ids)
    if batch[len(batch) - 1].terminated:
        return SamplingBatch(batch, "terminated")
    else:
        return SamplingBatch(batch, "batch_limit")


def show(label, batch):
    assert isinstance(batch,SamplingBatch), "需要返回SamplingBatch对象"
    assert isinstance(batch.transitions,list) and batch.transitions
    assert all(isinstance(row,Transition) for row in batch.transitions)
    print(f"{label}: 条数={len(batch.transitions)} stop_reason={batch.stop_reason} "
          f"rewards={[row.reward for row in batch.transitions]} "
          f"last_terminated={batch.transitions[-1].terminated}")


def main():
    try:
        game = ScriptedGame()
        first = collect_batch(game,4,2)
        show("第一批",first)
        assert len(first.transitions)==2 and first.stop_reason=="batch_limit"
        assert not first.transitions[-1].terminated and game.actions==[4,4]
        second = collect_batch(game,0,2)
        show("同一局第二批",second)
        assert len(second.transitions)==1 and second.stop_reason=="terminated"
        assert second.transitions[0].reward==0.5 and second.transitions[0].terminated
        assert first.transitions[-1].next_observation==second.transitions[0].observation
        assert game.actions==[4,4,0], "不能重复执行或越过终止"
        assert [row.reward for row in first.transitions+second.transitions]==[0.5,0.0,0.5]
        for budget in (3,5):
            fresh = ScriptedGame()
            batch = collect_batch(fresh,9,budget)
            show(f"新局预算{budget}",batch)
            assert len(batch.transitions)==3 and batch.stop_reason=="terminated"
            assert fresh.actions==[9,9,9]
            assert all(row.action==9 for row in batch.transitions)
        one = collect_batch(ScriptedGame(),3,1)
        assert len(one.transitions)==1 and one.stop_reason=="batch_limit"
        assert one.transitions[0].action==3 and not one.transitions[0].terminated
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。完整接口在文件开头。")
        return
    print("练习通过：采样停止原因明确且批次可接续；未创建训练器。")


if __name__ == "__main__":
    main()
