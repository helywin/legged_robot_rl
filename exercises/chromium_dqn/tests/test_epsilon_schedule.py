"""检查探索衰减边界、全局计数、重置与经验池环回。"""
from pathlib import Path
from dataclasses import replace
import sys
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import TrainConfig, epsilon_at, train_loop, vector_train_loop
from tensor_replay import TensorReplay
from agent import Agent
from task import ResetResult, StepResult


class TinyTask:
    def reset(self, seed: int) -> ResetResult:
        self.tick = 0
        return ResetResult([0., 0.], {})

    def step(self, action: int) -> StepResult:
        self.tick += 1
        return StepResult([float(self.tick), 0.], 0., False, self.tick == 2, {})


class ScheduleChecks(unittest.TestCase):
    def test_default_schedule_and_invalid_settings(self) -> None:
        config = TrainConfig(max_updates=1001)
        for completed, expected in [(0, 1.), (255, 1.), (256, 1.),
                                    (656, .525), (1056, .05), (1256, .05)]:
            self.assertAlmostEqual(epsilon_at(config, completed), expected)
        for kwargs in ({'epsilon_end': -.1}, {'epsilon_start': 2.},
                       {'epsilon_end': .8, 'epsilon_start': .2},
                       {'epsilon_decay_fraction': 0}, {'epsilon_decay_fraction': 1.1}):
            with self.assertRaises(ValueError):
                replace(config, **kwargs)

    def test_budget_scaling_and_prefill_only(self) -> None:
        for updates, span in [(1, 1), (501, 400), (1001, 800), (40000, 32000)]:
            config = TrainConfig(max_updates=updates)
            self.assertEqual(config.total_decisions, 255 + updates)
            self.assertEqual(config.epsilon_decay_decisions, span)
            if updates == 1:
                self.assertEqual(epsilon_at(config, config.total_decisions - 1), 1.)
            else:
                self.assertEqual(epsilon_at(config, config.total_decisions - 1), .05)

    def test_single_and_vector_use_same_schedule_through_wrap_and_reset(self) -> None:
        sequences: list[list[float]] = []
        for count in (1, 4):
            config = replace(TrainConfig(), num_envs=count, learning_starts=3,
                             capacity=8, batch_size=2, max_updates=20,
                             epsilon_decay_fraction=.5)
            online, target = nn.Linear(2, 3), nn.Linear(2, 3)
            agent = Agent(online, target, torch.optim.SGD(online.parameters(), lr=.001))
            replay = TensorReplay(8, 2)
            records: list[dict] = []
            if count == 1:
                train_loop(TinyTask(), agent, replay, config, records.append)
            else:
                vector_train_loop([TinyTask() for _ in range(count)], agent, replay, config, records.append)
            values = [r['epsilon'] for r in records]
            self.assertEqual(values[:4], [1.] * 4)
            self.assertAlmostEqual(values[8], .525)
            self.assertEqual(values[13:], [.05] * (len(values) - 13))
            self.assertEqual(len(records), 22)
            sequences.append(values)
        self.assertEqual(sequences[0], sequences[1])


if __name__ == '__main__':
    unittest.main()
