"""第087课：真实游戏连续分批采样，唯一变化量为批次大小。

仓库根目录依次运行：
  .venv/bin/python experiments/2026-09-10-chromium-sampling-batches/run.py --batch-size 2
  .venv/bin/python experiments/2026-09-10-chromium-sampling-batches/run.py --batch-size 3

每次1个同步GUI、共同初始化1个IDLE tick，再执行总共6次RIGHT，每次1tick，
每步停留0.4秒、批间1秒、最后2秒，0次网络训练。两次仅改变每批收几条。
先预测：大小2分3批，每批2条；大小3分2批，每批3条。若未提前真正终止，
两次各6条、各6tick，原因均batch_limit；批次间相邻观察应首尾相接。
同一运行内始终复用一个GameClient，batch_limit不重开游戏。
学习者观察终端batch/rows/tick和窗口连续运动，比较分批是否改变总动作数。
没有公共seed，不要求两个独立进程的完整随机状态一致。早期分数增量可能全0，
是没有新增得分，不是采样失败。只验证采样链，不训练策略或证明最终观察充分。
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chromium_rl import Action, GameClient
from exercises.sampling_stop_reason import collect_batch
from exercises.game_resource_observation import build_with_resources


class NativeGameAdapter:
    """将旧采样器的整数动作转换为原生客户端要求的Action，状态直接转发。"""
    def __init__(self, client):
        self.client = client
        self.actual_ticks = 0
        self.last_tick = None

    def snapshot(self):
        return self.client.snapshot()

    def step(self, action_id):
        result = self.client.step(Action(action_id), ticks=1)
        self.actual_ticks += result.actual_ticks
        if self.last_tick is not None:
            assert result.episode_tick == self.last_tick + result.actual_ticks
        self.last_tick = result.episode_tick
        print(f"  tick={result.episode_tick} action={action_id} "
              f"world_x={result.snapshot.player.position[0]:.6f} "
              f"score={result.snapshot.player.score:g}", flush=True)
        time.sleep(0.4)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(2,3), required=True)
    args = parser.parse_args()
    with GameClient(synchronous=True) as client:
        initial = client.step(Action.IDLE)
        if initial.terminated:
            raise RuntimeError("初始化已真正结束，不能继续分批采样")
        game = NativeGameAdapter(client)
        game.last_tick = initial.episode_tick
        all_rows = []
        batch_index = 0
        while len(all_rows) < 6:
            batch_index += 1
            budget = min(args.batch_size, 6-len(all_rows))
            print(f"batch={batch_index} budget={budget}", flush=True)
            batch = collect_batch(game, int(Action.RIGHT), budget)
            assert 0 < len(batch.transitions) <= budget
            if all_rows:
                assert all_rows[-1].next_observation == batch.transitions[0].observation
            all_rows.extend(batch.transitions)
            assert all(a.next_observation == b.observation for a,b in zip(all_rows,all_rows[1:]))
            assert all(row.action == int(Action.RIGHT) for row in batch.transitions)
            if batch.stop_reason == "terminated":
                assert batch.transitions[-1].terminated
                print("真正结束，保留最后一条并停止；本次不算完整6步对照。", flush=True)
                break
            assert batch.stop_reason == "batch_limit" and len(batch.transitions) == budget
            assert not any(row.terminated for row in batch.transitions)
            print(f"rows={len(batch.transitions)} stop_reason={batch.stop_reason} "
                  f"rewards={[row.reward for row in batch.transitions]}", flush=True)
            before_wait = client.snapshot()
            time.sleep(1)
            assert client.snapshot() == before_wait, "批间等待不应推进游戏"
        assert len(all_rows) == game.actual_ticks
        final = client.snapshot()
        assert all_rows[-1].next_observation == build_with_resources(final)
        total_reward = sum(row.reward for row in all_rows)
        score_delta = (final.player.score-initial.snapshot.player.score)/100.0
        assert abs(total_reward-score_delta) < 1e-9
        print(f"总计: batches={batch_index} rows={len(all_rows)} "
              f"actual_ticks={game.actual_ticks} reward_sum={total_reward} "
              f"final_world_x={final.player.position[0]:.6f}", flush=True)
        print("批间观察已核对衔接；窗口保留2秒。", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止，未将中断记为完整对照通过。")
