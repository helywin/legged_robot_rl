"""090：实现经验存储与批量转换，连接Task与已经通过的Agent.update。

只修改add、sample、make_batch三个TODO，可以加辅助函数；禁止导入旧实现。
运行（仓库根）：.venv/bin/python exercises/chromium_dqn/check_replay.py
当前脚手架未完成时友好提示，不启动游戏、不声称训练通过。

Transition是一条经验，Batch是一批经验的张量表示，两者职责不同。
Transition观察使用tuple[float, ...]，由训练循环用tuple(observation)从Task的
列表创建；这样未来修改原列表不会改变旧经验。frozen禁止重绑字段，tuple禁止
修改其中某个元素；本接口要求元组中元素都是float，不接收可变list观察。
自然终止与外部截断分别存储；下一观察必须来自动作后同一局，不能填reset结果。

Replay(capacity, seed)：self._items是按加入先后排列的list[Transition]，
self._rng是独立的random.Random，只用于采样，不与动作探索共享随机状态。
add(row) -> None：追加一条；超过capacity时移除最早一条。不要删最新经验。
sample(batch_size) -> list[Transition]：数量必须是正整数且不超过当前数量，
否则ValueError。使用self._rng随机抽取，无放回、不删除库存、不改变库存顺序。
允许使用self._rng.sample，不能把“总取最后几条”当作随机回放。

make_batch(rows) -> Batch：非空经验列表转成CPU张量，各行仍一一对应。
observations与next_observations用float32，[B, observation_size]；
actions用int64，[B]；rewards用float32，[B]；两个结束标志用bool，[B]。
可用torch.tensor(二维或一维列表, dtype=...)。不要把actions变成[B,1]，
因为Agent.update内部已经负责unsqueeze。诊断允许2维观察，真实任务为62维。

手算容量例：capacity=3，依次加入A、B、C、D，库存应为B、C、D。
随机抽2条可能得到[D,B]：批次第0行的观察、动作、奖励、结束标志都必须来自D，
第1行都来自B。采样后库存仍为B、C、D。抽样顺序不是时间推进顺序。

成功条件：容量淘汰正确、采样无放回且不删库存、张量形状/类型正确，并能把
实际抽出的经验交给已有Agent.update完成参数更新。这还不是持续游戏训练。
"""
from dataclasses import dataclass
import random
import torch
from agent import Batch


@dataclass(frozen=True)
class Transition:
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool
    truncated: bool


class Replay:
    def __init__(self, capacity: int, seed: int = 7) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('capacity必须是正整数')
        self.capacity: int = capacity
        self._items: list[Transition] = []
        self._rng: random.Random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, row: Transition) -> None:
        if len(self._items) >= self.capacity:
            self._items.remove(self._items[0])
        self._items.append(row)

    def sample_batch(self, batch_size: int) -> Batch:
        return make_batch(self.sample(batch_size))

    def sample(self, batch_size: int) -> list[Transition]:
        if type(batch_size) is not int or batch_size <=0:
            raise ValueError("batch_size数值错误")

        return self._rng.sample(self._items, batch_size)


def make_batch(rows: list[Transition]) -> Batch:
    if len(rows) == 0:
        raise ValueError("不接受空数据")

    observations = torch.tensor(
        [trans.observation for trans in rows], dtype=torch.float32
    )
    actions = torch.tensor(
        [trans.action for trans in rows], dtype=torch.int64
    )
    rewards = torch.tensor(
        [trans.reward for trans in rows], dtype=torch.float32
    )
    next_observations = torch.tensor(
        [trans.next_observation for trans in rows], dtype=torch.float32
    )
    terminateds = torch.tensor(
        [trans.terminated for trans in rows], dtype=torch.bool
    )
    truncateds = torch.tensor(
        [trans.truncated for trans in rows], dtype=torch.bool
    )
    return Batch(
        observations,
        actions,
        rewards,
        next_observations,
        terminateds,
        truncateds
    )
