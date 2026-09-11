"""教师边界检查；人工环境不算40局原生评测证据。"""
from dataclasses import replace
import math
from evaluate import evaluate_episode, summarize, EpisodeResult
from task import ResetResult, StepResult


class DiagnosticTask:
    def __init__(self, reason: str):
        self.reason = reason
        self.count = 0
        self.resets = 0

    def reset(self, seed: int) -> ResetResult:
        self.resets += 1
        self.count = 0
        return ResetResult([1.0], dict(score=0.0, lives_counter=4))

    def step(self, action: int) -> StepResult:
        assert self.count < 2, '结束后不应再step'
        assert action == self.count + 1, '必须按新的观察选择动作'
        self.count += 1
        last = self.count == 2
        return StepResult([float(self.count+1)], 1.0, last and self.reason != 'decision_limit',
                          last and self.reason == 'decision_limit',
                          dict(score=float(100*self.count), lives_counter=3 if last else 5,
                               actual_ticks=2 if last else 5, end_reason=self.reason if last else None))


def checks() -> None:
    for reason in ('hero_dead','level_over','decision_limit'):
        task = DiagnosticTask(reason)
        result = evaluate_episode(task, lambda obs: int(obs[0]), 201)
        assert result.seed == 201 and result.decisions == 2
        assert math.isclose(result.simulated_seconds, 0.14)
        assert result.score == 200.0 and result.total_reward == 2.0
        assert result.observed_life_decreases == 2, '先加命再损命，不能只算首尾差'
        assert result.end_reason == reason and task.resets == 1
    base = EpisodeResult(201, 10, 1.0, 0.0, 0.0, 1, 'hero_dead')
    rows = [base, replace(base, seed=202, score=100.0, simulated_seconds=2.0,
                         observed_life_decreases=0, end_reason='level_over'),
            replace(base, seed=203, score=900.0, simulated_seconds=3.0,
                    observed_life_decreases=2, end_reason='decision_limit')]
    result = summarize(rows)
    assert result.episodes == 3
    assert math.isclose(result.mean_score, 1000.0/3)
    assert result.median_score == 100.0 and result.mean_seconds == 2.0
    assert result.mean_life_decreases == 1.0
    for ratio in (result.completion_rate,result.death_rate,result.truncation_rate):
        assert math.isclose(ratio,1/3)
    try:
        summarize([])
        raise AssertionError('空结果应拒绝')
    except ValueError:
        pass
    print('评测循环与汇总检查通过；下一步--run才执行双方各20局真实评测。')
