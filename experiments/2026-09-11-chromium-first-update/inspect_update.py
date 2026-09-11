"""教师诊断：以原配置重现首个真实游戏更新，调用学习者现有代码。"""
from pathlib import Path
import sys
import json
from dataclasses import replace
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'exercises/chromium_dqn'))
from agent import Agent, Batch, UpdateStats
from network import QNetwork
from replay import Replay
from runtime import Runtime
from task import GameTask
from train import TrainConfig, train_loop

SOURCE = ROOT / 'exercises/chromium_dqn/runs/train-06fbbc3a5a1c4ff78afc7ffabb1586c1'


class InspectedAgent(Agent):
    def update(self, batch: Batch) -> UpdateStats:
        # 以下诊断值不参与优化；更新仍由学习者Agent.update执行。
        with torch.no_grad():
            all_q = self.online(batch.observations)
            chosen = all_q.gather(1, batch.actions[:, None]).squeeze(1)
            next_best = self.target(batch.next_observations).max(dim=1).values
            targets = batch.rewards + self.gamma * next_best * (~batch.terminated).float()
            squares = (chosen - targets).square()
        index = int(squares.argmax())
        action = int(batch.actions[index])
        bias = self.online.layers[-1].bias
        old_bias = float(bias[action].detach())
        target_before = {key: value.clone() for key, value in self.target.state_dict().items()}
        outputs = []
        def capture(module, inputs, output):
            output.retain_grad()
            outputs.append(output)
        hook = self.online.register_forward_hook(capture)
        try:
            result = super().update(batch)
        finally:
            hook.remove()
        output = outputs[0]
        bias_gradient = float(bias.grad[action])
        detail = dict(
            batch_size=len(batch.actions), row=index, action=action,
            reward=float(batch.rewards[index]), terminated=bool(batch.terminated[index]),
            truncated=bool(batch.truncated[index]), prediction=float(chosen[index]),
            next_max_q=float(next_best[index]), gamma=self.gamma, target=float(targets[index]),
            error=float(chosen[index]-targets[index]), squared_error=float(squares[index]),
            squared_error_sum=float(squares.sum()), loss=result.loss,
            selected_q_gradient=float(output.grad[index, action]),
            analytic_selected_q_gradient=float(2*(chosen[index]-targets[index])/len(batch.actions)),
            selected_bias_before=old_bias, selected_bias_gradient=bias_gradient,
            same_action_rows=int((batch.actions==action).sum()),
            learning_rate=self.optimizer.param_groups[0]['lr'],
            selected_bias_after=float(bias[action].detach()),
            q_graph_node=type(output.grad_fn).__name__,
            target_unchanged=all(torch.equal(value,target_before[key]) for key,value in self.target.state_dict().items()),
            target_grad_none=all(p.grad is None for p in self.target.parameters()),
        )
        with torch.no_grad():
            detail['next_forward_prediction'] = float(self.online(batch.observations)[index, action])
        self.detail = detail
        return result


def main():
    original_config = json.loads((SOURCE/'config.json').read_text())
    original_config.pop('max_decisions', None)  # 历史总决策上限；本诊断仅复现第一次更新。
    config = replace(TrainConfig(**original_config), max_updates=1)
    torch.set_num_threads(1)
    torch.manual_seed(config.network_seed)
    online, target = QNetwork(), QNetwork()
    agent = InspectedAgent(online,target,torch.optim.SGD(online.parameters(),lr=config.learning_rate),
                           config.gamma,config.exploration_seed)
    with Runtime() as runtime:
        summary = train_loop(GameTask(runtime,config.episode_limit,task_version='task-v1'),agent,
                             Replay(config.capacity,config.replay_seed),config,lambda row: None)
    original = next(json.loads(line)['loss'] for line in (SOURCE/'steps.jsonl').read_text().splitlines()
                    if json.loads(line)['updates'] == 1)
    report = dict(agent.detail, decisions=summary.decisions, original_first_loss=original,
                  first_loss_exact_match=agent.detail['loss']==original)
    destination = ROOT/'exercises/chromium_dqn/runs/first-update-diagnostic.json'
    destination.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
