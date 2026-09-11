"""教师性能对照：1/4/8个无窗口环境，各3次4000更新，不作策略效果验收。

命令：.venv/bin/python experiments/2026-09-11-chromium-training-throughput/benchmark.py
顺序运行，避免各组争抢CPU；生成检查点和原始日志只存项目runs。
"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'exercises/chromium_dqn'))
from train import TrainConfig, run_native


def main() -> None:
    rows: list[dict[str, object]] = []
    for repeat in range(3):
        for count in (1,4,8):
            output = StringIO()
            with redirect_stdout(output):
                run_native(TrainConfig(max_updates=4000, num_envs=count, task_version='task-v2-powerups', epsilon_start=.2, epsilon_end=.2))
            line = next(line for line in output.getvalue().splitlines() if line.startswith('输出目录：'))
            directory = Path(line.split('：',1)[1].strip())
            report: dict[str, object] = json.loads((directory/'summary.json').read_text())
            report.update(repeat=repeat, directory=str(directory))
            rows.append(report)
            print(count, repeat, report['elapsed_seconds'], flush=True)
    medians = {str(count): statistics.median(float(row['elapsed_seconds']) for row in rows if row['num_envs']==count)
               for count in (1,4,8)}
    destination = ROOT/'exercises/chromium_dqn/runs/throughput-benchmark.json'
    destination.write_text(json.dumps(dict(rows=rows,median_seconds=medians),indent=2))
    print(medians)


if __name__ == '__main__': main()
