"""预分配CPU经验池：NumPy共享存储写入，PyTorch独立批次采样。"""
import random
from numpy import ndarray
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
        # numpy视图与CPU张量共享内存；赋值直接转换到目标dtype，避免六次张量写入派发。
        self._write_views: tuple[ndarray, ...] = tuple(tensor.numpy() for tensor in (
            self.observations, self.actions, self.rewards, self.next_observations,
            self.terminated, self.truncated,
        ))

    def __len__(self) -> int:
        return self._size

    def add(self, row: Transition) -> None:
        index: int = self._next
        for view, value in zip(self._write_views, (
            row.observation, row.action, row.reward, row.next_observation,
            row.terminated, row.truncated,
        )):
            view[index] = value
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
