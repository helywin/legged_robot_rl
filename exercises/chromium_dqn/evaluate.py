"""092：比较训练模型与随机策略（学习者实现两个核心函数）。

现在要做什么：完成evaluate_episode与summarize，先--check，再--run。
 .venv/bin/python exercises/chromium_dqn/evaluate.py --check
 .venv/bin/python exercises/chromium_dqn/evaluate.py --run

问题：冻结的训练模型，在同样任务规则下能否比随机动作获得更好成绩？
固定比较：同一个policy.pt、task-v1、18动作、每动作5tick、每局250决策上限。
模型关闭探索，随机策略均匀选择0..17（也允许移动开火组合），都不更新参数。
双方各20局，当前默认游戏seed为30001..30020（早期实验为201..220）；随机动作每局用独立seed=100000+游戏seed。
相同种子便于配对，但不同动作会使游戏轨迹和后续随机消耗分叉，不承诺同一弹幕。
本轮使用headless，不依赖显示服务。40局最多10000决策（约50000原生tick），
预计秒级至几十秒，具体以elapsed_seconds为准。这是短回合对照，不是完整游戏验收。

教师提供：权重加载、策略选择回调、运行调度、输出路径与JSON保存。
你只写下面两个TODO，不修改检查器，也不需要重写神经网络或训练逻辑。

一、evaluate_episode(task, choose_action, seed) -> EpisodeResult
整体步骤：reset一次→按当前观察选动作→step→累计统计→结束时返回。
choose_action(observation)直接返回Python int，内部已处理模型/随机差异。
统计字段及来源：
 seed：传入种子；decisions：成功step次数；simulated_seconds：实际ticks总和×0.02，
 不能用决策次数×5替代，因为最后一步可能提前结束。
 score：最后result.info['score']；total_reward：本局奖励累加。
 observed_life_decreases：从reset的info['lives_counter']开始，每步累加
 max(0, 上一步生命计数-这一步生命计数)。每步更新比较基准，不能只比较首尾。
 这是快照可见的下降量，不是精确死亡事件数；同一步奖励生命可能抵消损失。
 end_reason：最后result.info['end_reason']（hero_dead、level_over或decision_limit）。
更新观察后继续；terminated或truncated为True即结束，不额外重开。
不要调用optimizer、update、sync_target或清空训练回放。

二、summarize(rows) -> Aggregate
拒绝空列表(ValueError)，汇总这组EpisodeResult：局数、平均得分、得分中位数、
平均模拟存活秒数、平均可见生命计数下降、单关完成率、自然死亡率、截断率。
三个比率按end_reason分别统计level_over、hero_dead、decision_limit，再除以局数。
可用statistics.mean/median，允许平均值为float，不把一局高分当成总体表现。
例：得分[0,100,900]，均值333.333，中位数100；中位数能帮助识别少数高分拉高均值。

成功条件：人工边界检查通过；双方20局结果落盘，确认评测前后模型参数不变。
模型赢、输、打平都可以完成有效评测，不能为了得到好结果中途修改种子/上限。
同一组种子用于后续反复调参会过拟合，后续正式验收需要另一组保留种子。
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import argparse
import hashlib
import json
import random
import statistics
import time
from uuid import uuid4
import torch
from task import observation_size_for
from network import QNetwork
from runtime import Runtime
from task import GameTask, TASK_VERSION, OBSERVATION_SIZE, action_count_for, TICKS


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    decisions: int
    simulated_seconds: float
    score: float
    total_reward: float
    observed_life_decreases: int
    end_reason: str
    events: dict[str, int | float] | None = None


@dataclass(frozen=True)
class Aggregate:
    episodes: int
    mean_score: float
    median_score: float
    mean_seconds: float
    mean_life_decreases: float
    completion_rate: float
    death_rate: float
    truncation_rate: float
    mean_events: dict[str, float] | None = None


def evaluate_episode(
    task: GameTask,
    choose_action: Callable[[list[float]], int],
    seed: int,
) -> EpisodeResult:
    """同一个单局循环服务两种策略；按顶部契约收集真实结果。"""
    reset_result = task.reset(seed)
    observation = reset_result.observation

    decisions: int = 0
    total_ticks: int = 0
    total_reward: float = 0.0
    lives: int = reset_result.info['lives_counter']
    observed_life_decreases: int = 0
    events: dict[str, int | float] | None = None

    while 1:
        action = choose_action(observation)
        step_result = task.step(action)
        decisions += 1
        # 游戏时间按实际推进量统计，最后一步可能不足约定的5tick。
        total_ticks += step_result.info['actual_ticks']
        total_reward += step_result.reward
        if 'events' in step_result.info:
            if events is None:
                events = {key: 0 for key in step_result.info['events']}
            for key, value in step_result.info['events'].items():
                events[key] += value
        current_lives: int = step_result.info['lives_counter']
        observed_life_decreases += max(0, lives - current_lives)
        # 增加生命时也更新基准，以免漏算随后发生的下降。
        lives = current_lives
        observation = step_result.observation

        if step_result.terminated or step_result.truncated:
            return EpisodeResult(
                seed,
                decisions,
                total_ticks * 0.02,
                step_result.info['score'],
                total_reward,
                observed_life_decreases,
                step_result.info['end_reason'],
                events,
            )



def summarize(rows: list[EpisodeResult]) -> Aggregate:
    """汇总所有回合；比率为0到1，不是百分数。"""
    if not rows:
        raise ValueError('评测结果不能为空')

    count: int = len(rows)
    scores: list[float] = [row.score for row in rows]
    mean_events: dict[str, float] | None = None
    available = [row.events for row in rows if row.events is not None]
    if len(available) == len(rows):
        mean_events = {key: float(statistics.mean(event[key] for event in available))
                       for key in available[0]}
    return Aggregate(
        mean_events=mean_events,
        episodes=count,
        mean_score=float(statistics.mean(scores)),
        median_score=float(statistics.median(scores)),
        mean_seconds=float(statistics.mean(row.simulated_seconds for row in rows)),
        mean_life_decreases=float(statistics.mean(row.observed_life_decreases for row in rows)),
        # 布尔比较的True计为1，统计对应结束原因的局数再除以总局数。
        completion_rate=sum(row.end_reason == 'level_over' for row in rows) / count,
        death_rate=sum(row.end_reason == 'hero_dead' for row in rows) / count,
        truncation_rate=sum(row.end_reason == 'decision_limit' for row in rows) / count,
    )


def run_comparison(checkpoint: Path, seed_start: int = 30001) -> None:
    if not 0 <= seed_start <= 4294967295 - 19:
        raise ValueError("评测起始种子须允许连续20个uint32种子")
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    action_count: int = action_count_for(saved['task_version'])
    for key, expected in dict(observation_size=observation_size_for(saved['task_version']),
                              action_count=action_count, ticks=TICKS).items():
        if saved.get(key) != expected:
            raise ValueError(f'检查点任务规格不匹配：{key}')
    network = QNetwork(saved['observation_size'], action_count)
    network.load_state_dict(saved['online'], strict=True)
    network.eval()
    if not all(torch.isfinite(p).all().item() for p in network.parameters()):
        raise ValueError('非有限网络参数')
    before = {name: p.detach().clone() for name, p in network.named_parameters()}
    seeds: list[int] = list(range(seed_start, seed_start + 20))
    destination = Path(__file__).parent / 'runs' / ('eval-' + uuid4().hex)
    destination.mkdir(parents=True)
    (destination/'config.json').write_text(json.dumps(dict(checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ("task.py", "network.py", "runtime.py", "evaluate.py")},
        game_seeds=seeds, random_action_seed_offset=100000, task_version=saved['task_version'],
        episode_limit=saved['config']['episode_limit'], ticks=TICKS, actions=action_count,
        native_version=saved['native_version']), indent=2))
    print('评测预算：模型/随机各20局，共40局；每局上限', saved['config']['episode_limit'], '决策；不更新参数。')
    print('输出目录：', destination)
    start = time.monotonic()
    groups: dict[str, list[EpisodeResult]] = {}
    def model_action(observation: list[float]) -> int:
        with torch.no_grad():
            values = network(torch.tensor([observation], dtype=torch.float32))
            return int(values.argmax(dim=1).item())
    with (destination/'episodes.jsonl').open('w') as log:
        for name in ('model', 'random'):
            groups[name] = []
            with Runtime(headless=True) as runtime:
                if runtime.implementation != saved['native_version']:
                    raise ValueError('原生行为版本不匹配')
                task = GameTask(runtime, saved['config']['episode_limit'], task_version=saved['task_version'])
                for seed in seeds:
                    rng = random.Random(100000+seed)
                    def random_action(observation: list[float]) -> int:
                        return rng.randrange(action_count)
                    chooser = model_action if name == 'model' else random_action
                    result = evaluate_episode(task, chooser, seed)
                    groups[name].append(result)
                    log.write(json.dumps(dict(policy=name, **asdict(result)), allow_nan=False)+'\n')
                    log.flush()
                    print(name, seed, result.score, result.end_reason)
    if any(not torch.equal(before[name], p) for name, p in network.named_parameters()):
        raise RuntimeError('评测期间参数发生变化')
    report = {name: asdict(summarize(rows)) for name, rows in groups.items()}
    report['elapsed_seconds'] = time.monotonic()-start
    report['parameters_unchanged'] = True
    (destination/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    print('结果已保存；还需根据得分、存活和结束分布作结论，不能只选最好的一局。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seed-start', type=int, default=30001, help='连续20局起始种子，默认30001；须避开训练开局')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.check:
            from check_evaluate import checks
            checks()
        else:
            if args.checkpoint is None:
                parser.error('--run必须明确指定--checkpoint，避免误用旧权重')
            run_comparison(args.checkpoint, args.seed_start)
    except NotImplementedError as error:
        print('评测实作尚未完成：', error)
