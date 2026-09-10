"""第089课：真实经验回放容量4/6对照，只有容量变化。

仓库根目录依次运行：
  .venv/bin/python experiments/2026-09-10-chromium-replay-capacity/run.py --capacity 4
  .venv/bin/python experiments/2026-09-10-chromium-replay-capacity/run.py --capacity 6

每次1个同步GUI，共同初始化1个IDLE tick，再6次RIGHT（每次1tick）。
分为先1条、再5条两批。每次入库后请求抽2条，固定抽样随机种子7。
第一批库存1不足2，应已保存但抽样结果为空；第二批按容量淘汰后随机抽2条。
每步等待0.3秒，批间1秒，最后2秒；0次网络更新，不生成模型或数据集。
编号1..6仅打印追踪，不加进观察。容量4保留3..6，容量6保留1..6。
观察抽样后保存编号是否不变，实际抽样条目是否属于所保留范围。
只有抽样种子固定，游戏无公共seed，跨进程原始坐标不要求逐字段相同。
未提前终止时才完成6条对照；提前终止保留实际结果，不继续同一已终止游戏。
"""
import argparse
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chromium_rl import Action, GameClient
from exercises.sampling_stop_reason import collect_batch
from exercises.game_replay_sampling import GameReplayBuffer, store_and_sample


class NativeGameAdapter:
    def __init__(self, client):
        self.client = client
        self.ticks = 0

    def snapshot(self):
        return self.client.snapshot()

    def step(self, action_id):
        result = self.client.step(Action(action_id), ticks=1)
        self.ticks += result.actual_ticks
        time.sleep(0.3)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, choices=(4,6), required=True)
    args = parser.parse_args()
    with GameClient(synchronous=True) as client:
        initial = client.step(Action.IDLE)
        if initial.terminated:
            raise RuntimeError("初始化已终止，不能开始本对照")
        game = NativeGameAdapter(client)
        buffer = GameReplayBuffer(args.capacity)
        rng = random.Random(7)
        all_rows = []
        numbers = {}
        for count in (1,5):
            batch = collect_batch(game, int(Action.RIGHT), count)
            for row in batch.transitions:
                numbers[id(row)] = len(all_rows)+1
                all_rows.append(row)
            native_before = client.snapshot()
            sampled = store_and_sample(buffer, batch, 2, rng)
            assert client.snapshot() == native_before, "入库/抽样不应推进原生游戏"
            assert all(a is b for a,b in zip(buffer.snapshot(),all_rows[-args.capacity:]))
            assert len(buffer) == min(len(all_rows),args.capacity)
            saved_ids = [numbers[id(row)] for row in buffer.snapshot()]
            sampled_ids = [numbers[id(row)] for row in sampled]
            print(f"本批{len(batch.transitions)}条 stop_reason={batch.stop_reason} "
                  f"保存编号={saved_ids} 抽样编号={sampled_ids}", flush=True)
            for row in sampled:
                print(f"  经验{numbers[id(row)]}: obs_x={row.observation[0]:.6f} "
                      f"action={row.action} reward={row.reward} "
                      f"next_obs_x={row.next_observation[0]:.6f} terminated={row.terminated}", flush=True)
            if batch.stop_reason == "terminated":
                print("任务提前真正结束，本次不算完整6条容量对照。", flush=True)
                return
            assert batch.stop_reason == "batch_limit" and len(batch.transitions)==count
            assert len(sampled) == (0 if len(buffer)<2 else 2)
            assert len({id(row) for row in sampled})==len(sampled)
            assert all(any(row is stored for stored in buffer.snapshot()) for row in sampled)
            time.sleep(1)
        assert len(all_rows)==6 and game.ticks==6
        assert all(a.next_observation==b.observation for a,b in zip(all_rows,all_rows[1:]))
        before = buffer.snapshot()
        again = buffer.sample(2,rng)
        assert buffer.snapshot()==before, "随机抽样不删除原经验"
        print("再次抽样编号=",[numbers[id(row)] for row in again],
              "抽样后保存编号=",[numbers[id(row)] for row in buffer.snapshot()], flush=True)
        print(f"检查通过：6条真实经验，容量{args.capacity}，实际{game.ticks}tick；未更新网络。", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止，未将中断记为对照通过。")
