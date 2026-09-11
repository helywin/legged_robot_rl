"""090最后一段：加载你训练的策略，亲手实现一局GUI回放。

目标：从保存的权重作决策，让你看见训练后的飞机行为。这里只推理，不更新。
只实现play_episode的TODO；加载、规格校验、绘图节奏与关闭由教师提供。
整体顺序：reset → 看观察选最大Q值动作 → step → 显示 → 判断结束。

输入接口：
 task: GameTask，reset(seed).observation获取开局观察，step(action)得到StepResult。
 network: QNetwork，已严格加载policy.pt并eval；不需要再创建目标网络或优化器。
 seed: int，本局游戏种子；与训练种子区分，默认101是新开局，不是训练录像。
 show: Callable[[StepResult],None]，每步调用，教师封装负责显示当前画面并适度放慢。
 返回PlaybackSummary，记录决策数、累计任务奖励、最后原始总分和结束原因。

步骤提示：
 1. reset后取初始观察，决策数与累计奖励从0开始。
 2. 用no_grad把一条观察编码成float32的[1,62]张量，送入network。
 3. 沿动作维argmax，转Python int，交给task.step。不加入随机探索。
 4. 决策数加1、累计奖励增加，调用show(result)，用新观察继续。
 5. terminated或truncated为True即停止，不再reset或step；返回统计。
 最后分数取result.info['score']，结束原因取result.info['end_reason']。

注意：eval()切换层的训练/评测行为，不负责关闭梯度；no_grad()才关闭本段求导记录。
当前网络只有Linear/ReLU，所以eval不改变其数值计算，但仍明确表达推理模式。
显示函数不更新游戏，不修改网络；游戏只被task.step推进。
不要求本次通关。请观察主要动作、是否开火、是否长期顶边和最后为何结束。

运行（仓库根；--check只检查加载，--run才执行你实现的GUI循环）：
 .venv/bin/python exercises/chromium_dqn/play.py --check
 .venv/bin/python exercises/chromium_dqn/play.py --run
可用--checkpoint PATH指定其他本项目权重，--seed 101设置开局种子。
单局最多250决策、每次5tick，显示后按模拟时长暂停，最多约25秒加计算开销。
保存文件不是录像：每一步动作会由新开局的实际观察重新计算。
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import argparse
import json
import time
import torch
from network import QNetwork
from runtime import Runtime
from task import GameTask, StepResult, TASK_VERSION, OBSERVATION_SIZE, ACTION_COUNT, TICKS


@dataclass(frozen=True)
class PlaybackSummary:
    decisions: int
    total_reward: float
    final_score: float
    end_reason: str


def play_episode(
    task: GameTask,
    network: QNetwork,
    seed: int,
    show: Callable[[StepResult], None],
) -> PlaybackSummary:
    """冻结策略玩一局，按顶部五步完成；不要进行参数更新。"""
    observation = torch.tensor(
        task.reset(seed).observation,
        dtype=torch.float32
    ).unsqueeze(0)
    decisions = 0
    total_reward = 0
    final_score = ""
    end_reason = ""

    while 1:
        with torch.no_grad():
            q_values = network.forward(observation)
            best_action = q_values.argmax(dim=1).item()
            result = task.step(int(best_action))
            show(result)
            observation = torch.tensor(
                result.observation,
                dtype=torch.float32
            ).unsqueeze(0)
            decisions += 1
            total_reward += result.reward
            final_score = result.info['score']
            end_reason = result.info['end_reason']
            if result.truncated or result.terminated:
                break;

    return PlaybackSummary(
        decisions,
        total_reward,
        final_score,
        end_reason
    )


def main() -> None:
    default_checkpoint = Path(__file__).parent / 'runs/train-06fbbc3a5a1c4ff78afc7ffabb1586c1/policy.pt'
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=default_checkpoint)
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    expected = dict(task_version=TASK_VERSION, observation_size=OBSERVATION_SIZE,
                    action_count=ACTION_COUNT, ticks=TICKS)
    for name, value in expected.items():
        if saved.get(name) != value:
            raise ValueError(f'任务规格不兼容：{name}，保存值={saved.get(name)}，当前值={value}')
    network = QNetwork()
    network.load_state_dict(saved['online'], strict=True)
    if not all(torch.isfinite(p).all().item() for p in network.parameters()):
        raise ValueError('保存参数包含非有限值')
    network.eval()
    print('权重与任务规格检查通过：', args.checkpoint)
    if args.check:
        return
    initial = {name: parameter.detach().clone() for name, parameter in network.named_parameters()}
    with Runtime() as runtime:
        if runtime.implementation != saved['native_version']:
            raise ValueError('原生行为版本不同，请先核实兼容性')
        task = GameTask(runtime, max_decisions=saved['config']['episode_limit'])
        def show(result: StepResult) -> None:
            runtime.render()
            time.sleep(result.info['actual_ticks'] * 0.02)
        summary = play_episode(task, network, args.seed, show)
    if any(not torch.equal(initial[name], parameter) for name, parameter in network.named_parameters()):
        raise RuntimeError('回放期间参数发生变化')
    print(json.dumps(asdict(summary), indent=2))
    print('回放参数保持不变；单局行为不代表独立评测或通关能力。')


if __name__ == '__main__':
    try:
        main()
    except NotImplementedError as error:
        print('回放实作尚未完成：', error)
