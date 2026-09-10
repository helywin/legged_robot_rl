"""第088课：把采样批次接到经验回放保存与抽样（仅标准库）。

只补全store_and_sample里的TODO。缓冲区复用033课deque保存/淘汰/抽样机制，
这里明确使用083的向量Transition类型，不导入055中依赖torch的示例。

输入与接口
==========
buffer：本文件GameReplayBuffer，外层创建一次，在多批之间复用。
batch：086的SamplingBatch，batch.transitions是这次采到的经验列表。
sample_size：这次希望随机抽取的条数，不是采样批次长度，也不是缓冲区容量。
rng：外层创建的random.Random对象，本函数不要重建或反复重设种子。

buffer.add(row)：加入一条完整Transition；满了自动淘汰最旧的。
len(buffer)：当前已保存条数。
buffer.sample(sample_size, rng)：随机抽取互不重复的完整经验，返回list，不删除原数据。
buffer.snapshot()：仅供观察/检查，返回当前保存内容。

要实现的真实连接
================
先将batch.transitions按原顺序逐条add，不要把整个列表作为一条经验加入。
全部加入后，如果当前保存条数少于sample_size，返回空列表[]，表示暂不抽样；
否则调用buffer.sample并返回其列表。不要每批新建缓冲区，不按stop_reason清空。
terminated经验也应保存，batch_limit只是采样批次结束，不清理历史经验。
返回的是抽样结果list[Transition]，不是缓冲区对象，不是新增经验的总列表。
一条经验的五个字段一起抽取，不能分别打乱observation/action/reward。

固定数据与可见预期
==================
六条教学经验用动作前x观察0.0、0.1、0.2、0.3、0.4、0.5标识，容量为4。
第一次输入第一条，sample_size=2，保存1条但返回[]。
第二次输入后五条，旧的0.0/0.1被淘汰，留下[0.2,0.3,0.4,0.5]。
用random.Random(7)首次抽2条，得到[0.4,0.2]；缓冲区仍保留原来的4条。
编号只供教学追踪，不作为新增观察字段。随机种子只固定抽样顺序，不固定游戏。

运行（仓库根目录）
==================
    .venv/bin/python -m exercises.game_replay_sampling

成功条件：不足时先保存但不抽样；多批共享内容；超容量按时间淘汰；抽样不删除；
保存终止经验、不按批次清空；完整经验字段不拆散。
这是可接入087真实SamplingBatch的连接函数，本次离线轨迹验证数据处理，不训练
网络。抽样后的不同经验可以不相邻，但每条内部的前后状态必须保持原样。
"""
from collections import deque
import random
from dataclasses import replace

from exercises.game_resource_observation import PlayerSample, SnapshotSample
from exercises.game_step_transition import Transition, StepResultSample, make_transition
from exercises.sampling_stop_reason import SamplingBatch


class GameReplayBuffer:
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity必须是正整数")
        self._rows: deque[Transition] = deque(maxlen=capacity)

    def __len__(self):
        return len(self._rows)

    def add(self, row: Transition):
        self._rows.append(row)

    def snapshot(self):
        return tuple(self._rows)

    def sample(self, sample_size: int, rng: random.Random):
        if type(sample_size) is not int or not 1 <= sample_size <= len(self._rows):
            raise ValueError("抽样数量必须为正且不超过已保存数量")
        return rng.sample(list(self._rows), sample_size)


def store_and_sample(buffer: GameReplayBuffer, batch: SamplingBatch,
                     sample_size: int, rng: random.Random) -> list[Transition]:
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("sample_size必须是正整数")
    # 先逐条保存本批经验，再根据实际库存决定返回[]或随机抽样结果。
    for trans in batch.transitions:
        buffer.add(trans)

    if sample_size <= 0 or sample_size > len(buffer):
        return []

    
    return buffer.sample(sample_size, rng)


def labels(rows):
    return [row.observation[0] for row in rows]


def main():
    player = PlayerSample((0.0,-3.0,0.0),(0.0,0.0),-500.0,500.0,4,score=1000.0)
    frames = [SnapshotSample(replace(player,position=(float(i),-3.0,0.0),score=1000.0+50*i),())
              for i in range(7)]
    rows = [make_transition(frames[i],4,StepResultSample(frames[i+1],i==5)) for i in range(6)]
    buffer = GameReplayBuffer(4)
    rng = random.Random(7)
    try:
        first = store_and_sample(buffer,SamplingBatch(rows[:1],"batch_limit"),2,rng)
        print("第一批后: 保存=", labels(buffer.snapshot()), "抽到=", labels(first))
        assert first==[] and buffer.snapshot()==tuple(rows[:1]), "不足时也必须保存，并返回空列表"
        second = store_and_sample(buffer,SamplingBatch(rows[1:],"terminated"),2,rng)
        print("第二批后: 保存=", labels(buffer.snapshot()), "抽到=", labels(second))
        assert isinstance(second,list) and len(second)==2
        assert buffer.snapshot()==tuple(rows[2:]), "按时间淘汰，不因批次或终止清空"
        assert all(a is b for a,b in zip(buffer.snapshot(),rows[2:])), "应保留完整原经验对象"
        assert second[0] is rows[4] and second[1] is rows[2], "检查是否复用传入随机对象、完整随机抽样"
        before = buffer.snapshot()
        # 没有新数据也可重复抽取，之前抽到的经验没有被消费掉。
        again = store_and_sample(buffer,SamplingBatch([],"batch_limit"),2,rng)
        print("再次抽样: 抽到=", labels(again), "缓冲区条数=",len(buffer))
        assert buffer.snapshot()==before and len(again)==2
        assert len({id(row) for row in again})==2
        assert all(any(row is stored for stored in before) for row in again)
        assert buffer.snapshot()[-1].terminated, "最后一步经验也要保留"
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。完整接口与返回值在文件开头。")
        return
    print("练习通过：经验跨批保存、有限容量淘汰、抽样不删除；尚未更新网络。")


if __name__ == "__main__":
    main()
