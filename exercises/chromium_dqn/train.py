"""090第三段：亲手实现真实DQN训练循环。

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


当前任务已升级为task-v2-powerups（110维），旧task-v1权重仍可独立评测回放。
下一次道具观察对照：运行--run --max-updates 4000，与既有4000更新的62维模型比较。
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
from pathlib import Path
from typing import Callable
import argparse
import json
import shlex
import time
from uuid import uuid4
import torch
from agent import Agent
from network import QNetwork
from replay import Replay, Transition, make_batch
from runtime import Runtime
from task import GameTask, TASK_VERSION, OBSERVATION_SIZE, ACTION_COUNT, TICKS


@dataclass(frozen=True)
class TrainConfig:
    max_updates: int = 500
    episode_limit: int = 250
    capacity: int = 10000
    batch_size: int = 32
    learning_starts: int = 256
    target_sync_every: int = 100
    epsilon: float = 0.2
    gamma: float = 0.99
    learning_rate: float = 0.001
    game_seed: int = 31
    network_seed: int = 7
    replay_seed: int = 7
    exploration_seed: int = 11


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
    replay: Replay,
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
        epsilon = 1.0 if len(replay) < config.learning_starts else config.epsilon
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
            status = agent.update(make_batch(replay.sample(config.batch_size)))
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
            loss = step_loss
        ))

    return TrainingSummary(
        decisions,
        episodes_finished,
        updates,
        total_reward,
        last_loss
    )

def run_native(config: TrainConfig) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(config.network_seed)
    destination = Path(__file__).parent / 'runs' / ('train-' + uuid4().hex)
    destination.mkdir(parents=True)
    (destination / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    online: QNetwork = QNetwork(OBSERVATION_SIZE, ACTION_COUNT)
    target: QNetwork = QNetwork(OBSERVATION_SIZE, ACTION_COUNT)
    optimizer: torch.optim.Optimizer = torch.optim.SGD(online.parameters(), lr=config.learning_rate)
    agent = Agent(online, target, optimizer, config.gamma, config.exploration_seed)
    replay = Replay(config.capacity, config.replay_seed)
    before = {name: parameter.detach().clone() for name, parameter in online.named_parameters()}
    print('任务：', TASK_VERSION, '观察维数：', OBSERVATION_SIZE)
    print('预算：1个游戏，CPU；', config.max_updates, '次更新；每动作', TICKS, 'tick；每局最多', config.episode_limit, '次决策')
    print('输出目录：', destination)
    start = time.monotonic()
    with (destination / 'steps.jsonl').open('w') as log, Runtime() as runtime:
        def record(row: dict[str, object]) -> None:
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
        task = GameTask(runtime, max_decisions=config.episode_limit)
        # 首次GUI显示开局；循环会显式reset同一种子，避免依赖这个演示状态。
        runtime.reset(config.game_seed)
        runtime.render()
        summary = train_loop(task, agent, replay, config, record)
        native_version = runtime.implementation
    changed = any(not torch.equal(before[name], value) for name, value in online.named_parameters())
    report = dict(asdict(summary), parameter_changed=changed, elapsed_seconds=time.monotonic()-start,
                  native_version=native_version)
    (destination / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    if summary.updates <= 0 or not changed:
        raise RuntimeError('没有实际更新或参数变化，不能将本次运行记为训练通过')
    torch.save(dict(task_version=TASK_VERSION, observation_size=OBSERVATION_SIZE,
                    action_count=ACTION_COUNT, ticks=TICKS, config=asdict(config),
                    native_version=native_version, online=online.state_dict()), destination / 'policy.pt')
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
    args = parser.parse_args()
    if args.max_updates <= 0:
        parser.error('--max-updates必须为正整数')
    try:
        if args.check:
            from check_train import checks
            checks()
        elif args.run:
            run_native(TrainConfig(max_updates=args.max_updates))
        else:
            parser.print_help()
    except NotImplementedError as error:
        print('训练循环尚未实现：', error)
