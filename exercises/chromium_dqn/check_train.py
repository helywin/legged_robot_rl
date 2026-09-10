"""教师循环诊断：模拟环境用于检查边界，不是原生训练证据。"""
from dataclasses import replace
from unittest.mock import patch
import torch
from torch import nn
from agent import Agent
from replay import Replay
from task import ResetResult, StepResult
from train import TrainConfig, train_loop


class DiagnosticTask:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.count = 0
        self.ready = False

    def reset(self, seed: int) -> ResetResult:
        self.seeds.append(seed)
        self.count = 0
        self.ready = True
        return ResetResult([float(len(self.seeds)), 0.0], {})

    def step(self, action: int) -> StepResult:
        assert self.ready and type(action) is int and 0 <= action < 3
        self.count += 1
        ended = self.count == 2
        self.ready = not ended
        return StepResult([float(len(self.seeds)), float(self.count)], 0.1, False, ended, {})


def checks() -> None:
    config = replace(TrainConfig(), learning_starts=3, batch_size=2, max_updates=2,
                     max_decisions=20, target_sync_every=2, capacity=8, episode_limit=2)
    for cap, expected_decisions, expected_updates in ((20, 4, 2), (3, 3, 1)):
        torch.manual_seed(7)
        online = nn.Linear(2, 3)
        target = nn.Linear(2, 3)
        agent = Agent(online, target, torch.optim.SGD(online.parameters(), lr=0.001))
        replay = Replay(8, seed=7)
        task = DiagnosticTask()
        records: list[dict[str, object]] = []
        with patch.object(agent, 'sync_target', wraps=agent.sync_target) as sync:
            result = train_loop(task, agent, replay, replace(config, max_decisions=cap), records.append)
            assert sync.call_count == (2 if cap == 20 else 1)
        assert result.decisions == expected_decisions and result.updates == expected_updates, result
        assert result.episodes_finished == expected_decisions // 2
        assert abs(result.total_reward - expected_decisions * 0.1) < 1e-6
        assert result.last_loss is not None
        assert task.seeds == [config.game_seed, config.game_seed + 1]
        assert len(records) == len(replay) == expected_decisions
        rows = sorted(replay.sample(len(replay)), key=lambda row: (row.observation[0], row.observation[1]))
        assert rows[1].next_observation == (1.0, 2.0), '上一局末观察被reset结果覆盖'
        assert rows[2].observation == (2.0, 0.0)
        if cap == 3:
            assert not rows[-1].terminated and not rows[-1].truncated, '训练预算不能伪造回合结束'
        assert [row['updates'] for row in records] == [0, 0, 1, 2][:expected_decisions]
        assert [row['epsilon'] for row in records] == [1.0, 1.0, 1.0, config.epsilon][:expected_decisions]
        assert [row['episode'] for row in records] == [1, 1, 2, 2][:expected_decisions]
        print('循环诊断通过：', result)
    print('人工环境验证了真实网络更新与调度；真实游戏训练仍需--run。')
