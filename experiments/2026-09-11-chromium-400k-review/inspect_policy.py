"""教师只读诊断：明确加载40万更新模型，记录动作、资源与奖励；不训练。
运行：.venv/bin/python experiments/2026-09-11-chromium-400k-review/inspect_policy.py
"""
from collections import Counter
from pathlib import Path
import json
import sys
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'exercises/chromium_dqn'))
from runtime import Runtime
from network import QNetwork
from task import encode_observation, compute_reward, TASK_VERSION


def main() -> None:
    checkpoint = ROOT/'exercises/chromium_dqn/runs/train-f0de1e18d5194f349877ec548b0eb98b/policy.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['task_version'] == 'task-v2-powerups'
    torch.set_num_threads(1)
    model = QNetwork(saved['observation_size']); model.load_state_dict(saved['online']); model.eval()
    reports: list[dict[str,object]] = []
    with Runtime(headless=True) as game:
        for seed in (30001,30002,30003):
            before = game.reset(seed)
            rows: list[dict[str,object]] = []
            for decision in range(1,251):
                with torch.no_grad():
                    values = model(torch.tensor([encode_observation(before)],dtype=torch.float32))[0]
                    action = int(values.argmax())
                result = game.step(action,5)
                after = result.snapshot
                gone = [dict(id=p.id, type=p.type, y=p.position[1]) for p in before.powerups
                        if p.id not in {q.id for q in after.powerups}]
                rows.append(dict(decision=decision,action=action,position=after.player.position[:2],
                                 score=after.player.score,reward=compute_reward(before,after,saved['task_version']),
                                 lives=after.player.lives_counter,
                                 life_drop=max(0,before.player.lives_counter-after.player.lives_counter),
                                 ammo=after.player.ammo_stock,shields=after.player.shields,
                                 vanished_powerups=gone, q_max=float(values.max()), q_min=float(values.min())))
                before = after
                if result.terminated:break
            summary = dict(seed=seed,decisions=len(rows),score=rows[-1]['score'],
                           end_reason=result.termination_reason or 'decision_limit',
                           actions=dict(Counter(r['action'] for r in rows)),
                           edge_decisions=sum(abs(r['position'][0])>=9.5 or abs(r['position'][1])>=7 for r in rows),
                           ammo_ever_positive=any(any(a>0 for a in r['ammo']) for r in rows),
                           reward_events=[r for r in rows if r['reward']],
                           life_events=[r for r in rows if r['life_drop']])
            reports.append(dict(summary=summary, steps=rows))
            print(json.dumps(summary),flush=True)
    (ROOT/'exercises/chromium_dqn/runs/400k-policy-diagnostic.json').write_text(json.dumps(reports,indent=2))


if __name__ == '__main__': main()
