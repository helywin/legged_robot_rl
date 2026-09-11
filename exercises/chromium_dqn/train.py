"""090第三段：亲手实现真实DQN训练循环。

无窗口训练（默认1个环境；--num-envs 8开启8个独立游戏进程并发采样）：
  .venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 40000 --num-envs 8
只有一个总训练预算max_updates；N个环境共享模型/经验池，不是每个各训练40000次。
每新增一条达到预填门槛的经验仍更新一次；并行不是通过少更新获得虚假的加速。

增加整体训练预算：
  .venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 40000
训练只限制更新次数。预填256条后每决策更新一次，40000更新需要40255决策。
每局250决策后重开，不中断累计更新。上述命令会从头训练，不是续训。


你已完成Task、Replay、网络和Agent。本次只实现train_loop的TODO，把它们连接。
教师提供配置、启动/关闭、日志和检查点文件样板；不替你写循环。
从仓库根目录运行：
  .venv/bin/python exercises/chromium_dqn/train.py --check
  .venv/bin/python exercises/chromium_dqn/train.py --run
--check执行独立的循环诊断；--run才启动真实游戏与训练。无参数显示说明。

目标：一段完整训练，直到max_updates次更新完成。
一个游戏进程、CPU、每动作5tick；首版SGD学习率0.001，不引入新的优化算法。
预填充256条，批量32，之后每决策更新一次；默认500次更新，总决策数只统计、不限制。
每局250次决策上限，通常第755次决策达到500次更新（256那步开始更新）。
预填充阶段epsilon=1.0，之后0.2；预填充结束以本步选动作前库存量判断。
这些是流程验收预算，短回合会限制策略学习范围，不作为游戏能力评测规则。

只修改train_loop函数体，可加小辅助函数；不要复制旧训练循环。
参数类型与用途：
 task: GameTask，reset(seed)->ResetResult，step(action)->StepResult。
 agent: Agent，act(observation,epsilon)->int，update(Batch)->UpdateStats，sync_target()->None。
 replay: Replay，add(Transition)->None，sample(n)->list[Transition]，len(replay)->int。
 config: TrainConfig，见下面字段；调用前已经验证合法。
 record: Callable[[dict[str,object]],None]，写入一条JSON日志；不用自己管理文件。
 返回TrainingSummary，包含本次决策次数、已结束回合数、本次更新次数、累计奖励、最后损失。
 首版要求传入全新Agent和空Replay；此函数不是续训接口。

循环顺序提示（保留整个循环由你实现）：
 1. 首次agent.sync_target()；task.reset(config.game_seed)取得当前观察。
 2. 判断更新预算是否用尽；若没有，依据当前库存决定epsilon，agent.act选动作。
 3. task.step执行一次。用旧观察、动作、本步奖励、新观察及两个结束标志创建Transition。
    观察列表转换tuple；先把本步经验存入Replay，不能先reset覆盖下一观察。
 4. 本次决策数和累计奖励增加；如果库存达到learning_starts，则sample→make_batch→update。
    每满target_sync_every次更新显式同步目标网络；按更新次数，不按游戏tick同步。
 5. 本步结束时已结束回合数加1；更新当前观察。若还要继续且回合已结束，用
    game_seed+已结束回合数重开，否则直接使用刚得到的新观察。
    不清空Replay或Agent；预算耗尽只停止训练，不额外伪造terminated/truncated。
 6. 每步调用record，字段必须包含decision、updates、episode、reward、terminated、
    truncated、epsilon、loss。episode是本步所属的回合编号，从1开始；loss为本步
    更新损失，未更新时为None。最后返回TrainingSummary；last_loss是最近一次更新损失。

手算调度：若预填充3、批量2、更新上限2，前两步不更新；第三步存入后第一次更新，
第四步第二次更新并停。只执行4次决策，不再多走第5步。
若第2步回合截断，第2条经验仍保存第2步末观察；第3步才使用新局初始观察。

成功条件：诊断检查通过；真实运行产生非零更新、至少两个回合衔接和参数变化，
保存配置/日志/可加载权重。模型好坏稍后评测；检查点只保存推理所需信息，暂非续训状态。


当前默认任务为task-v3-events（110维，事件奖励），旧task-v1权重仍可独立评测回放。
当前奖励对照：--task-version task-v3-events与task-v2-powerups均为110维，只改变奖励。
观察压缩对照：--task-version task-v4-no-motion为84维，奖励与v3相同。
当前默认v7为36维、逐次命中奖励；探索率在预填后剩余决策预算的前80%从1.0降至0.05。
只新增道具输入及必要输入层维数，奖励、动作和其余超参数不变。

以下为094第一轮预算实验的历史说明（当时为task-v1，勿当作新版基线）：
问题：500次更新是否不足？唯一变化量max_updates：500→4000。
保持TrainConfig其他默认值，包括5000决策上限和每局250决策上限。
从同种子全新初始化，不加载旧policy.pt；旧权重仅供对照。
亲手运行：
  .venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 4000
预期4255决策、4000次更新，耗时预计几十秒（以实际输出为准）。
然后执行输出的评测命令，与092旧模型的20局结果比较。
成功条件：配置只差max_updates，实际完成预算，冻结新权重完成评测，
记录得分中位数、存活时间和死亡/截断/过关局数；不要求必然提升。
详细记录：experiments/2026-09-11-chromium-training-budget/README.md。
"""
from dataclasses import asdict, dataclass
from collections import deque
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Callable
import argparse
import hashlib
import json
import shlex
import time
import math
import sys
from uuid import uuid4
import torch
from agent import Agent
from network import QNetwork
from network_health import inspect_network
from replay import Replay, Transition
from tensor_replay import TensorReplay
from runtime import Runtime
from task import GameTask, ResetResult, StepResult, TASK_VERSION, HIT_TASK_VERSION, TRAIN_TASK_VERSIONS, observation_size_for, action_count_for, FIVE_ACTION_TASK_VERSION, TICKS


@dataclass(frozen=True)
class TrainConfig:
    max_updates: int = 500
    update_backend: str = 'eager'
    loss_kind: str = 'huber'
    num_envs: int = 1
    task_version: str = FIVE_ACTION_TASK_VERSION
    episode_limit: int = 1000
    capacity: int = 10000
    batch_size: int = 32
    learning_starts: int = 256
    target_sync_every: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05  # 全局决策衰减终点，不因开新局重置
    epsilon_decay_fraction: float = 0.8  # 剩余采样预算前80%衰减，最后20%保持终值
    gamma: float = 0.99
    learning_rate: float = 0.001
    game_seed: int = 31
    network_seed: int = 7
    replay_seed: int = 7
    exploration_seed: int = 11


    def __post_init__(self) -> None:
        for name in ('max_updates', 'num_envs', 'episode_limit', 'capacity', 'batch_size',
                     'learning_starts', 'target_sync_every'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name}必须是正整数')
        if not 0 <= self.gamma <= 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('gamma须在[0,1]，learning_rate须为有限正数')
        if not self.batch_size <= self.learning_starts <= self.capacity:
            raise ValueError('要求batch_size <= learning_starts <= capacity')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('要求0 <= epsilon_end <= epsilon_start <= 1')
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError('epsilon_decay_fraction必须在(0, 1]内')

    @property
    def total_decisions(self) -> int:
        # 第learning_starts条经验即可完成第一次更新。
        return self.learning_starts - 1 + self.max_updates

    @property
    def epsilon_decay_decisions(self) -> int:
        remaining: int = max(0, self.total_decisions - self.learning_starts)
        return max(1, math.ceil(remaining * self.epsilon_decay_fraction))


def epsilon_at(config: TrainConfig, decisions_before_action: int) -> float:
    """输入动作执行前已完成的累计决策数；reset和经验池覆盖都不重启衰减。

    前learning_starts次动作完全随机。此后从start线性降到end，
    衰减跨度由本次剩余总决策预算×epsilon_decay_fraction推导，
    不是独立步数上限。N环境共用全局序号；预算小到只有预填时保持全随机。
    """
    if decisions_before_action < config.learning_starts:
        return 1.0
    elapsed: int = decisions_before_action - config.learning_starts
    if elapsed >= config.epsilon_decay_decisions:
        return config.epsilon_end
    fraction: float = elapsed / config.epsilon_decay_decisions
    return max(config.epsilon_end,
               config.epsilon_start + (config.epsilon_end - config.epsilon_start) * fraction)


@dataclass(frozen=True)
class TrainingSummary:
    decisions: int
    episodes_finished: int
    updates: int
    total_reward: float
    last_loss: float | None


def train_loop(
    task: GameTask,
    agent: Agent,
    replay: Replay | TensorReplay,
    config: TrainConfig,
    record: Callable[[dict[str, object]], None],
) -> TrainingSummary:
    """ 按顶部的完整流程连接各模块，返回本次训练统计。"""
    agent.sync_target()
    observation = task.reset(config.game_seed).observation
    decisions = 0
    episodes_finished = 0
    episode: int = episodes_finished + 1
    updates = 0
    total_reward = 0.0
    last_loss: float | None = None
    step_loss: float | None = None
    need_reset = False
    while 1:
        if updates >= config.max_updates:
            break
        if need_reset:
            observation = task.reset(
                            config.game_seed + episodes_finished
                        ).observation
            need_reset = False
        epsilon = epsilon_at(config, decisions)
        action = agent.act(observation, epsilon)
        step_result = task.step(action)
        decisions += 1
        total_reward += step_result.reward
        episode = episodes_finished + 1
        
        transition = Transition(tuple(observation), action, step_result.reward, tuple(step_result.observation), step_result.terminated, step_result.truncated)
        replay.add(transition)
        if step_result.truncated or step_result.terminated:
            episodes_finished += 1
            need_reset = True
        else:
            observation = step_result.observation

        step_loss = None
        if len(replay) >= config.learning_starts:
            status = agent.update(replay.sample_batch(config.batch_size))
            updates += 1
            last_loss = status.loss
            step_loss = status.loss

            
            if updates % config.target_sync_every == 0:
                agent.sync_target()

        record(dict(
            decision = decisions,
            updates = updates,
            episode = episode,
            reward = step_result.reward,
            terminated = step_result.terminated,
            truncated = step_result.truncated,
            epsilon = epsilon,
            loss = step_loss,
            events = step_result.info.get('events'),
        ))

    return TrainingSummary(
        decisions,
        episodes_finished,
        updates,
        total_reward,
        last_loss
    )

def vector_train_loop(
    tasks: list[GameTask], agent: Agent, replay: TensorReplay, config: TrainConfig,
    record: Callable[[dict[str, object]], None],
) -> TrainingSummary:
    """N个原生进程并发step，共享策略与经验池；每条可学习经验仍更新一次。"""
    agent.sync_target()
    count: int = len(tasks)
    local_episodes: list[int] = [0] * count
    observations: list[list[float]] = [[] for _ in tasks]
    needs_reset: list[bool] = [True] * count
    decisions: int = 0
    updates: int = 0
    finished: int = 0
    total_reward: float = 0.0
    last_loss: float | None = None
    # 每个Runtime只允许一个未完成请求；按env顺序收集结果以免线程完成顺序影响训练。
    with ThreadPoolExecutor(max_workers=count) as workers:
        while updates < config.max_updates:
            remaining: int = max(0, config.learning_starts - 1 - len(replay)) + config.max_updates - updates
            active: int = min(count, remaining)
            resets: dict[int, Future[ResetResult]] = {i: workers.submit(tasks[i].reset, config.game_seed + i + local_episodes[i] * count)
                      for i in range(active) if needs_reset[i]}
            for i, future in resets.items():
                observations[i] = future.result().observation
                needs_reset[i] = False
            epsilons: list[float] = [epsilon_at(config, decisions + i) for i in range(active)]
            actions: list[int] = agent.act_batch(observations[:active], epsilons)
            pending: list[Future[StepResult]] = [workers.submit(tasks[i].step, actions[i]) for i in range(active)]
            for i, future in enumerate(pending):
                result = future.result()
                replay.add(Transition(tuple(observations[i]), actions[i], result.reward,
                                      tuple(result.observation), result.terminated, result.truncated))
                observations[i] = result.observation
                decisions += 1
                total_reward += result.reward
                step_loss: float | None = None
                if len(replay) >= config.learning_starts:
                    last_loss = step_loss = agent.update(replay.sample_batch(config.batch_size)).loss
                    updates += 1
                    if updates % config.target_sync_every == 0:
                        agent.sync_target()
                record(dict(decision=decisions, updates=updates, env_id=i,
                            episode=local_episodes[i]+1, reward=result.reward,
                            terminated=result.terminated, truncated=result.truncated,
                            epsilon=epsilons[i], loss=step_loss, events=result.info.get('events')))
                if result.terminated or result.truncated:
                    needs_reset[i] = True
                    local_episodes[i] += 1
                    finished += 1
    return TrainingSummary(decisions, finished, updates, total_reward, last_loss)


class TrainingProgress:
    """标准库进度条：终端原行刷新，重定向时低频输出完整行。"""

    def __init__(self, total: int, learning_starts: int) -> None:
        self.total: int = total
        self.learning_starts: int = learning_starts
        self.started: float = time.monotonic()
        self.last_render: float = self.started
        self.interactive: bool = sys.stderr.isatty()
        self.width: int = 0
        self.recent_rewards: deque[float] = deque(maxlen=1000)
        self.reward_sum: float = 0.0
        self.update(0, 0, force=True)

    def update(self, updates: int, decisions: int, force: bool = False,
               reward: float | None = None, epsilon: float = 1.0) -> None:
        # 每条经验都计入窗口；显示节流不能导致遗漏奖励。
        if reward is not None:
            if len(self.recent_rewards) == self.recent_rewards.maxlen:
                self.reward_sum -= self.recent_rewards[0]
            self.recent_rewards.append(reward)
            self.reward_sum += reward
        now: float = time.monotonic()
        if not force and now - self.last_render < (0.5 if self.interactive else 5.0):
            return
        self.last_render = now
        elapsed: float = now - self.started
        rate: float = updates / elapsed if elapsed > 0 else 0.0
        fraction: float = updates / self.total
        filled: int = int(20 * fraction)
        bar: str = '=' * filled + '-' * (20 - filled)
        eta: str = f'{(self.total - updates) / rate:.0f}s' if rate > 0 else '--'
        phase: str = (f'预填经验 {min(decisions, self.learning_starts)}/{self.learning_starts}'
                      if updates == 0 else '训练')
        mean_reward: str = (f'{self.reward_sum / len(self.recent_rewards):+.4f}'
                            if self.recent_rewards else '--')
        line: str = (f'{phase} [{bar}] {fraction:6.1%} 更新 {updates}/{self.total}'
                     f' | 决策 {decisions} | {rate:.0f} 更新/s | 已用 {elapsed:.0f}s | 剩余 {eta}'
                     f' | 近{len(self.recent_rewards)}决策均奖 {mean_reward} | ε {epsilon:.4f}')
        if self.interactive:
            self.width = max(self.width, len(line))
            print('\r' + line.ljust(self.width), end='', file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.interactive:
            print(file=sys.stderr, flush=True)


def run_native(config: TrainConfig) -> None:
    if config.loss_kind not in ('mse', 'huber'):
        raise ValueError('loss_kind必须是mse或huber')
    if config.update_backend not in ('eager', 'scripted'):
        raise ValueError('更新后端必须为eager或scripted')
    if config.task_version not in TRAIN_TASK_VERSIONS:
        raise ValueError('训练入口只支持v2至v7任务')
    if config.num_envs <= 0 or config.max_updates <= 0:
        raise ValueError('环境数与更新预算必须为正')
    if not 0 < config.batch_size <= config.learning_starts <= config.capacity:
        raise ValueError('要求批量大小 <= 预填量 <= 经验容量，且均为正')
    observation_size: int = observation_size_for(config.task_version)
    action_count: int = action_count_for(config.task_version)
    torch.set_num_threads(1)
    torch.manual_seed(config.network_seed)
    destination = Path(__file__).parent / 'runs' / ('train-' + uuid4().hex)
    destination.mkdir(parents=True)
    (destination / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in ('train.py', 'task.py', 'agent.py', 'fast_update.py',
                                  'network.py', 'runtime.py', 'tensor_replay.py', 'network_health.py')}
    (destination / 'source_sha256.json').write_text(json.dumps(source_hashes, indent=2))
    online: QNetwork = QNetwork(observation_size, action_count)
    target: QNetwork = QNetwork(observation_size, action_count)
    optimizer: torch.optim.Optimizer = torch.optim.SGD(online.parameters(), lr=config.learning_rate)
    backend_start: float = time.monotonic()
    if config.update_backend == 'scripted':
        from fast_update import ScriptedSGDAgent
        agent = ScriptedSGDAgent(online, target, optimizer, config.gamma, config.exploration_seed, config.loss_kind)
    else:
        agent = Agent(online, target, optimizer, config.gamma, config.exploration_seed, config.loss_kind)
    backend_init_seconds: float = time.monotonic() - backend_start
    replay = TensorReplay(config.capacity, observation_size, config.replay_seed)
    before = {name: parameter.detach().clone() for name, parameter in online.named_parameters()}
    print('任务：', config.task_version, '观察维数：', observation_size, '动作数：', action_count)
    print('更新后端：', config.update_backend, '损失：', config.loss_kind)
    print(f'探索：前{config.learning_starts}次决策全随机；之后{config.epsilon_start:g}→'
          f'{config.epsilon_end:g}，衰减跨度{config.epsilon_decay_decisions}次决策'
          f'（剩余预算的{config.epsilon_decay_fraction:.0%}）；总决策预算{config.total_decisions}')
    print('预算：', config.num_envs, '个无窗口游戏，CPU；', config.max_updates, '次更新；每动作', TICKS, 'tick；每局最多', config.episode_limit, '次决策')
    print('输出目录：', destination, flush=True)
    start = time.monotonic()
    with ExitStack() as stack:
        log = stack.enter_context((destination / 'steps.jsonl').open('w', buffering=1024*1024))
        runtimes: list[Runtime] = [stack.enter_context(Runtime(headless=True)) for _ in range(config.num_envs)]
        tasks: list[GameTask] = [GameTask(runtime, max_decisions=config.episode_limit, task_version=config.task_version) for runtime in runtimes]
        progress = TrainingProgress(config.max_updates, config.learning_starts)
        stack.callback(progress.close)
        health_log = stack.enter_context((destination / 'health.jsonl').open('w'))
        def save_checkpoint(name: str, updates: int) -> None:
            torch.save(dict(task_version=config.task_version, observation_size=observation_size,
                            action_count=action_count, ticks=TICKS, config=asdict(config),
                            native_version=runtimes[0].implementation, updates=updates, source_sha256=source_hashes,
                            online=online.state_dict()), destination / name)
        def record(row: dict[str, object]) -> None:
            log.write(json.dumps(row, allow_nan=False) + '\n')
            # 保留逐步日志，但不为每条经验强制刷新文件。
            if int(row['decision']) % 1000 == 0:
                log.flush()
            current_updates: int = int(row['updates'])
            if current_updates > 0 and (current_updates % 1000 == 0 or current_updates == config.max_updates):
                health = inspect_network(online, replay.observations[:min(len(replay), 256)])
                health_log.write(json.dumps(dict(updates=current_updates, **health)) + '\n')
                health_log.flush()
                if not health['finite'] or health['collapsed']:
                    save_checkpoint('diagnostic.pt', current_updates)
                    raise RuntimeError(f'第{current_updates}次更新网络失活或输出非有限，已停止训练；'
                                       f'查看{destination}/health.jsonl和diagnostic.pt')
            if current_updates > 0 and current_updates % 10000 == 0:
                save_checkpoint(f'policy-update-{current_updates:09d}.pt', current_updates)
            progress.update(int(row['updates']), int(row['decision']),
                            force=int(row['updates']) == config.max_updates, reward=float(row['reward']),
                            epsilon=float(row['epsilon']))
        try:
            if config.num_envs == 1:
                summary = train_loop(tasks[0], agent, replay, config, record)
            else:
                summary = vector_train_loop(tasks, agent, replay, config, record)
        except Exception as error:
            save_checkpoint('diagnostic.pt', agent.updates)
            (destination / 'failure.json').write_text(json.dumps(
                dict(updates=agent.updates, error=str(error)), ensure_ascii=False, indent=2))
            raise
        native_version = runtimes[0].implementation
    changed = any(not torch.equal(before[name], value) for name, value in online.named_parameters())
    report = dict(asdict(summary), parameter_changed=changed, elapsed_seconds=time.monotonic()-start,
                  native_version=native_version, num_envs=config.num_envs, headless=True,
                  update_backend=config.update_backend, backend_init_seconds=backend_init_seconds)
    report['updates_per_second'] = summary.updates / report['elapsed_seconds']
    report['decisions_per_second'] = summary.decisions / report['elapsed_seconds']
    (destination / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    if summary.updates <= 0 or not changed:
        raise RuntimeError('没有实际更新或参数变化，不能将本次运行记为训练通过')
    torch.save(dict(task_version=config.task_version, observation_size=observation_size,
                    action_count=action_count, ticks=TICKS, config=asdict(config),
                    native_version=native_version, source_sha256=source_hashes, updates=summary.updates,
                    online=online.state_dict()), destination / 'policy.pt')
    print(json.dumps(report, indent=2))
    print('已保存推理权重policy.pt；GUI加载回放和策略效果尚待验证。')
    print('评测本次权重：')
    print('.venv/bin/python exercises/chromium_dqn/evaluate.py --run --checkpoint '
          + shlex.quote(str(destination / 'policy.pt')))
    print('GUI回放本次权重：')
    print('.venv/bin/python exercises/chromium_dqn/play.py --run --checkpoint '
          + shlex.quote(str(destination / 'policy.pt')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--max-updates', type=int, default=500,
                        help='更新次数上限；094使用4000，其余训练配置保持默认')
    parser.add_argument('--num-envs', type=int, default=1, help='并行游戏进程数；共享一个模型和经验池，默认1')
    parser.add_argument('--loss-kind', choices=('mse', 'huber'), default='huber',
                        help='huber抑制大误差对更新的影响；mse用于旧版对照')
    parser.add_argument('--update-backend', choices=('eager', 'scripted'), default='eager',
                        help='eager默认教学版；scripted实验性CPU SGD编译加速')
    parser.add_argument('--task-version', choices=TRAIN_TASK_VERSIONS, default=FIVE_ACTION_TASK_VERSION,
                        help='v2分差奖励110维；v3事件奖励110维；v4事件奖励84维；v5另加受伤掉盾惩罚；v6同v5奖励36维；v7逐次子弹伤害奖励；v8同v7观察奖励、5动作固定开火（默认）')
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error('--num-envs必须为正整数')
    if args.max_updates <= 0:
        parser.error('--max-updates必须为正整数')
    try:
        if args.check:
            from check_train import checks
            checks()
        elif args.run:
            run_native(TrainConfig(max_updates=args.max_updates, num_envs=args.num_envs,
                                   task_version=args.task_version,
                                   update_backend=args.update_backend, loss_kind=args.loss_kind))
        else:
            parser.print_help()
    except NotImplementedError as error:
        print('训练循环尚未实现：', error)
