"""冻结权重诊断：真实状态下的隐藏层、Q值变化和动作分布，不训练。
.venv/bin/python exercises/chromium_dqn/diagnose.py --checkpoint PATH --seed 1
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import torch
from network import QNetwork
from network_health import inspect_network
from runtime import Runtime
from task import action_count_for, native_action_for, TICKS, observation_size_for, encode_task_observation, compute_reward


def diagnose(checkpoint: Path, seed: int) -> dict:
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    version: str = saved['task_version']
    action_count: int = action_count_for(version)
    if (saved['observation_size'] != observation_size_for(version)
            or saved['action_count'] != action_count or saved['ticks'] != TICKS):
        raise ValueError('权重任务规格不匹配')
    network = QNetwork(saved['observation_size'], action_count)
    network.load_state_dict(saved['online'])
    network.eval()
    if not all(torch.isfinite(p).all() for p in network.parameters()):
        raise ValueError('非有限参数')
    original = {name: value.clone() for name, value in network.state_dict().items()}
    rows: list[dict] = []
    observations: list[list[float]] = []
    with Runtime(headless=True) as runtime:
        if runtime.implementation != saved['native_version']:
            raise ValueError('原生行为版本不匹配')
        state = runtime.reset(seed)
        for decision in range(1, saved['config']['episode_limit'] + 1):
            observation = encode_task_observation(state, version)
            observations.append(observation)
            with torch.no_grad():
                values = network(torch.tensor([observation], dtype=torch.float32))[0]
            action = int(values.argmax())
            result = runtime.step(native_action_for(action, version), TICKS)
            rows.append(dict(decision=decision, action=action, q=values.tolist(),
                             position=result.snapshot.player.position[:2],
                             reward=compute_reward(state, result.snapshot, version)))
            state = result.snapshot
            if result.terminated:
                break
    discounted: float = 0.
    # 截断时这是有限回放段回报，不伪称完整无限时域回报。
    for row in reversed(rows):
        discounted = row['reward'] + saved['config']['gamma'] * discounted
        row['observed_discounted_return'] = discounted
    health = inspect_network(network, torch.tensor(observations, dtype=torch.float32))
    assert all(torch.equal(original[name], value) for name, value in network.state_dict().items())
    report = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  seed=seed, task_version=version, decisions=len(rows), health=health,
                  action_counts=dict(Counter(r['action'] for r in rows)),
                  first_q_max=max(rows[0]['q']), observed_discounted_return=discounted,
                  total_reward=sum(r['reward'] for r in rows),
                  end_reason=result.termination_reason if result.terminated else 'decision_limit',
                  final_events=state.events.__dict__ if state.events is not None else None,
                  parameters_unchanged=True)
    destination = Path(__file__).parent / 'runs' / ('diagnose-' + uuid4().hex)
    destination.mkdir(parents=True)
    (destination / 'summary.json').write_text(json.dumps(report, indent=2))
    (destination / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps(report, indent=2))
    print('诊断目录：', destination)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    diagnose(args.checkpoint, args.seed)
