"""训练用预分配经验池，避免每次更新把32条Python观察重新转成张量。"""
import random
import torch
from agent import Batch
from replay import Transition


class TensorReplay:
    def __init__(self, capacity: int, observation_size: int, seed: int = 7) -> None:
        if capacity <= 0 or observation_size <= 0:
            raise ValueError('容量与观察维数必须为正')
        self.capacity: int = capacity
        self._size: int = 0
        self._next: int = 0
        self._rng: random.Random = random.Random(seed)
        self.observations: torch.Tensor = torch.empty(capacity, observation_size, dtype=torch.float32)
        self.next_observations: torch.Tensor = torch.empty_like(self.observations)
        self.actions: torch.Tensor = torch.empty(capacity, dtype=torch.int64)
        self.rewards: torch.Tensor = torch.empty(capacity, dtype=torch.float32)
        self.terminated: torch.Tensor = torch.empty(capacity, dtype=torch.bool)
        self.truncated: torch.Tensor = torch.empty(capacity, dtype=torch.bool)

    def __len__(self) -> int:
        return self._size

    def add(self, row: Transition) -> None:
        index: int = self._next
        self.observations[index].copy_(torch.tensor(row.observation, dtype=torch.float32))
        self.next_observations[index].copy_(torch.tensor(row.next_observation, dtype=torch.float32))
        self.actions[index] = row.action
        self.rewards[index] = row.reward
        self.terminated[index] = row.terminated
        self.truncated[index] = row.truncated
        self._next = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_batch(self, batch_size: int) -> Batch:
        if type(batch_size) is not int or not 0 < batch_size <= self._size:
            raise ValueError('采样量须为正整数且不超过库存')
        logical: list[int] = self._rng.sample(range(self._size), batch_size)
        oldest: int = self._next if self._size == self.capacity else 0
        indices: torch.Tensor = torch.tensor([(i + oldest) % self.capacity for i in logical], dtype=torch.int64)
        # index_select返回独立批次；后续写入环形池不改变已有Batch。
        return Batch(*(tensor.index_select(0, indices) for tensor in (
            self.observations, self.actions, self.rewards, self.next_observations,
            self.terminated, self.truncated,
        )))
