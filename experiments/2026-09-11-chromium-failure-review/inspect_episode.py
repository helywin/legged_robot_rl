"""教师诊断，复现评测seed209并记录动作/位置；不修改策略或训练。"""
from pathlib import Path
import sys,json,collections
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'exercises/chromium_dqn'))
from network import QNetwork
from task import GameTask
from runtime import Runtime


def main():
    checkpoint=ROOT/'exercises/chromium_dqn/runs/train-06fbbc3a5a1c4ff78afc7ffabb1586c1/policy.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    torch.set_num_threads(1)
    model=QNetwork();model.load_state_dict(saved['online']);model.eval()
    rows=[]
    with Runtime() as runtime:
        task=GameTask(runtime,saved['config']['episode_limit'],task_version=saved['task_version'])
        state=task.reset(209)
        obs=state.observation
        previous_lives=state.info['lives_counter']
        while True:
            with torch.no_grad():
                q=model(torch.tensor([obs],dtype=torch.float32))[0]
                action=int(q.argmax())
            result=task.step(action)
            # 任务观察前两项为归一化世界坐标，还原用于位置诊断。
            x,y=result.observation[0]*10,result.observation[1]*7.5
            lives=result.info['lives_counter']
            rows.append(dict(decision=len(rows)+1,action=action,x=x,y=y,
                             reward=result.reward,score=result.info['score'],lives=lives,
                             life_drop=max(0,previous_lives-lives),actual_ticks=result.info['actual_ticks'],
                             end_reason=result.info['end_reason']))
            previous_lives=lives;obs=result.observation
            if result.terminated or result.truncated:break
    histogram=dict(collections.Counter(r['action'] for r in rows))
    report=dict(seed=209,decisions=len(rows),score=rows[-1]['score'],end_reason=rows[-1]['end_reason'],
                seconds=sum(r['actual_ticks'] for r in rows)*0.02,actions=histogram,
                edge_decisions=sum(abs(r['x'])>=9.5 or abs(r['y'])>=7 for r in rows),
                firing_decisions=sum(r['action']>=9 for r in rows),
                life_events=[r for r in rows if r['life_drop']],
                score_events=[r for r in rows if r['reward']!=0],first_steps=rows[:5])
    destination=ROOT/'exercises/chromium_dqn/runs/failure-seed209.json'
    destination.write_text(json.dumps(dict(summary=report,steps=rows),indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
