"""第067课：自己实现奖励函数，再做真实训练对照。

问题：原始奖励只鼓励存活，增加位置惩罚能否减少漂移？
唯一因素：是否使用位置惩罚；其余固定为本次基线的100000环境步、
100000回放容量、种子20260904及原有网络/更新参数。

允许修改：仅 position_reward 函数体。你需要把自然语言目标翻译成计算：
先用动作后的 x 除以位置边界2.4米，再平方，乘权重0.5，从原始奖励扣除。
平方同时表达左右对称、偏离越远惩罚越强。返回普通float，无需torch。
不要对奖励调用backward，不要修改环境终止条件、动作或折扣因子。

已有接口：environment_reward: float；next_observation: np.ndarray，
顺序[x, x_dot, theta, theta_dot]，x=next_observation[0]。
可用 float(...)、除法 /、平方 ** 2；常数已给出。
接线已经完成：函数返回值进入回放经验的reward，训练日志和冻结评估
仍然累计原始存活奖励。你的函数会在真正的环境循环里每步运行。

命令（仓库根目录）：
 .venv/bin/python experiments/2026-09-07-cartpole-position-reward/train.py baseline
 .venv/bin/python experiments/2026-09-07-cartpole-position-reward/train.py shaped

每次从头训练单环境100000步，约几十秒到数分钟；产物存入互不覆盖的
artifacts/cartpole-position-reward/<模式>-<时间戳>/，终端给出完整路径。
先完成函数再运行shaped；未完成时友好退出，不会启动训练。
每组保存.pt/.onnx/metrics.json/reward-experiment.json。

实作成功条件：完成真实训练，比较两组同一20个种子的原始存活步数。
效果假设：shaped平均步数提高且出轨比例降低；失败也要保留结果。
当前只测位置项，不保证消除所有震荡，更不保证理论稳定性。
GUI命令：
 .venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/watch.py <本轮online-network.pt路径>
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import gymnasium as gym
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2026-09-04-cartpole-dqn-smoke'))
from learner_train import CartPoleQNetwork, TrainingConfig, train_and_save

POSITION_LIMIT = 2.4
POSITION_WEIGHT = 0.5


def position_reward(environment_reward: float, next_observation: np.ndarray) -> float:
    """实现文件顶部解释的位置惩罚；不改变传入观察。"""
    x=next_observation[0]
    return 1 - (x/2.4)**2


def measure_behavior(checkpoint_path: Path, config: TrainingConfig) -> dict[str, object]:
    """同样20个开发种子，只推理；统计失败类型和整段轨迹偏移。"""
    network = CartPoleQNetwork()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    network.load_state_dict(checkpoint['online_network_state_dict'])
    network.eval()
    environment = gym.make('CartPole-v1')
    failures = {'track': 0, 'pole_angle': 0, 'time_limit_only': 0}
    positions: list[float] = []
    angles: list[float] = []
    try:
        for episode in range(config.evaluation_episodes):
            observation, _ = environment.reset(seed=config.seed + 10_000 + episode)
            while True:
                with torch.no_grad():
                    action = int(network(torch.as_tensor(observation, dtype=torch.float32)).argmax().item())
                observation, _, terminated, truncated, _ = environment.step(action)
                positions.append(float(observation[0]))
                angles.append(float(observation[2]))
                if terminated or truncated:
                    failures['track'] += int(abs(observation[0]) > 2.4)
                    failures['pole_angle'] += int(abs(observation[2]) > np.deg2rad(12))
                    failures['time_limit_only'] += int(truncated and not terminated)
                    break
    finally:
        environment.close()
    return {
        'end_causes': failures,  # 同一步可能同时违反两项，分别计数。
        'position_rms_m': float(np.sqrt(np.mean(np.square(positions)))),
        'angle_rms_deg': float(np.rad2deg(np.sqrt(np.mean(np.square(angles))))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['baseline', 'shaped'])
    mode = parser.parse_args().mode
    transform = position_reward if mode == 'shaped' else None
    if transform is not None:
        try:
            print('预览：x=1.2 m 时训练奖励 =', transform(1.0, np.array([1.2, 0., 0., 0.])))
        except NotImplementedError as error:
            print(error)
            return

    config = TrainingConfig(total_environment_steps=100_000, replay_capacity=100_000)
    directory = ROOT / 'artifacts/cartpole-position-reward' / (
        mode + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    )
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        'mode': mode, 'position_weight': POSITION_WEIGHT if transform else 0.0,
        'config': asdict(config), 'evaluation_reward': 'original environment survival reward',
        'status': 'running',
    }
    path = directory / 'reward-experiment.json'
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('单环境100000步，从头训练；本轮产物：', directory)
    metrics = train_and_save(config, reward_transform=transform, artifact_directory=directory)
    metadata['status'] = 'finished'
    metadata['behavior'] = measure_behavior(directory / 'online-network.pt', config)
    metadata['full_500_step_episodes'] = sum(
        value == 500 for value in metrics['trained_evaluation_returns']
    )
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('达到500步的回合数：', metadata['full_500_step_episodes'])
    print('冻结评估行为：', metadata['behavior'])


if __name__ == '__main__':
    main()
