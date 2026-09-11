"""并行调度与张量经验池契约，不以这些人工检查代替原生运行。"""
from pathlib import Path
import sys
import unittest
from dataclasses import replace
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import Replay, Transition
from tensor_replay import TensorReplay
from train import TrainConfig, vector_train_loop
from agent import Agent
from task import ResetResult, StepResult


class TinyTask:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.tick: int = 0

    def reset(self, seed: int) -> ResetResult:
        self.seeds.append(seed)
        self.tick = 0
        return ResetResult([float(seed), 0.0], {})

    def step(self, action: int) -> StepResult:
        assert self.tick < 2
        self.tick += 1
        return StepResult([float(self.seeds[-1]), float(self.tick)], 0.1, False, self.tick == 2, {})


class ParallelChecks(unittest.TestCase):
    def test_tensor_fifo_sampling_and_no_alias(self) -> None:
        old, fast = Replay(5, 7), TensorReplay(5, 2, 7)
        for i in range(12):
            row = Transition((float(i), 0.0), i % 3, float(i), (float(i), 1.0), False, i % 2 == 0)
            old.add(row); fast.add(row)
            if i >= 2:
                a,b = old.sample_batch(3), fast.sample_batch(3)
                for name in a.__dataclass_fields__:
                    self.assertTrue(torch.equal(getattr(a,name), getattr(b,name)), name)
        batch = fast.sample_batch(5)
        batch.observations.fill_(-100)
        self.assertTrue((fast.sample_batch(5).observations[:,0] >= 7).all())

    def test_exact_budget_and_independent_resets(self) -> None:
        for budget in (1,5,9):
            config = replace(TrainConfig(), num_envs=4, learning_starts=3, batch_size=2,
                             max_updates=budget, target_sync_every=2)
            tasks = [TinyTask() for _ in range(4)]
            model,target=nn.Linear(2,3),nn.Linear(2,3)
            agent=Agent(model,target,torch.optim.SGD(model.parameters(),lr=.001))
            replay=TensorReplay(30,2)
            records: list[dict[str,object]]=[]
            result=vector_train_loop(tasks,agent,replay,config,records.append)
            self.assertEqual(result.updates,budget)
            self.assertEqual(result.decisions,2+budget)
            self.assertEqual(len(records),2+budget)
            for i,task in enumerate(tasks):
                self.assertEqual(task.seeds,[31+i+4*j for j in range(len(task.seeds))])
            batch=replay.sample_batch(len(replay))
            self.assertTrue(torch.equal(batch.observations[:,0],batch.next_observations[:,0]),
                            '经验不得跨环境或使用reset后的下一观察')
            if budget==1:
                self.assertFalse(batch.truncated.any(),'总预算停止不得伪造截断')


if __name__ == '__main__': unittest.main()
