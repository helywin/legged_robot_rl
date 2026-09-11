"""固定训练配置比较普通/编译更新；顺序运行防止争抢CPU。

根目录运行：
.venv/bin/python experiments/2026-09-11-chromium-replay-throughput/benchmark.py
每组3次、每次10000更新、32环境、每局1000决策；先后顺序交替。
不仅比较耗时，还逐步核对全部经验日志和最终网络权重。失败立即报错。
采用当前工作区奖励；本轮测量时击毁系数为学习者修改的10，其他v5项保持不变。
"""
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import statistics
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'exercises/chromium_dqn'))
from train import TrainConfig, run_native


def main() -> None:
    rows: list[dict[str, object]] = []
    for repeat in range(3):
        pair: dict[str, Path] = {}
        for backend in (('eager', 'scripted') if repeat % 2 == 0 else ('scripted', 'eager')):
            output = StringIO()
            with redirect_stdout(output):
                run_native(TrainConfig(max_updates=10000, num_envs=32, episode_limit=1000,
                                       task_version='task-v5-shield-damage', update_backend=backend,
                                       epsilon_start=.2, epsilon_end=.2))
            directory = Path(next(line.split('：', 1)[1].strip() for line in output.getvalue().splitlines()
                                  if line.startswith('输出目录：')))
            pair[backend] = directory
            row = json.loads((directory / 'summary.json').read_text())
            row.update(repeat=repeat, path=str(directory))
            rows.append(row)
            print(backend, repeat, row['elapsed_seconds'], flush=True)
        assert (pair['eager'] / 'steps.jsonl').read_bytes() == (pair['scripted'] / 'steps.jsonl').read_bytes()
        left = torch.load(pair['eager'] / 'policy.pt', weights_only=True)['online']
        right = torch.load(pair['scripted'] / 'policy.pt', weights_only=True)['online']
        assert all(torch.equal(left[key], right[key]) for key in left)
    medians = {backend: statistics.median(row['elapsed_seconds'] for row in rows
                                         if row['update_backend'] == backend)
               for backend in ('eager', 'scripted')}
    result = dict(rows=rows, median_seconds=medians, exact_logs_and_weights=True,
                  source_sha256={name: hashlib.sha256((ROOT / 'exercises/chromium_dqn' / name).read_bytes()).hexdigest()
                                 for name in ('task.py', 'train.py', 'fast_update.py', 'tensor_replay.py')})
    destination = ROOT / 'exercises/chromium_dqn/runs/compiled-speed-comparison.json'
    destination.write_text(json.dumps(result, indent=2))
    print(medians, destination)


if __name__ == '__main__':
    main()
