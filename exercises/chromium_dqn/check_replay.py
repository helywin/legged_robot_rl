"""运行学习者回放/决策/批量更新；使用人工经验，不启动原生游戏。"""
import torch
from torch import nn
from replay import Replay, Transition, make_batch
from agent import Agent


def checks() -> None:
    rows = [Transition((float(i), 0.0), i % 3, float(i) / 10,
                       (float(i + 1), 0.0), i == 3, i == 2) for i in range(4)]
    replay = Replay(capacity=3, seed=7)
    for row in rows:
        replay.add(row)
    assert len(replay) == 3
    assert set(replay.sample(3)) == set(rows[1:])
    for invalid in (0, -1, 4, True):
        try:
            replay.sample(invalid)
            raise AssertionError('不合法的批量数量未拒绝')
        except ValueError:
            pass
    sampled = replay.sample(2)
    assert len(set(sampled)) == 2 and len(replay) == 3
    batch = make_batch(sampled)
    assert batch.observations.shape == (2, 2)
    for i, row in enumerate(sampled):
        assert batch.actions[i].item() == row.action
        assert abs(batch.rewards[i].item() - row.reward) < 1e-6
        assert tuple(batch.observations[i].tolist()) == row.observation
        assert tuple(batch.next_observations[i].tolist()) == row.next_observation
        assert batch.terminated[i].item() == row.terminated
        assert batch.truncated[i].item() == row.truncated
    assert batch.actions.dtype == torch.int64 and batch.actions.shape == (2,)
    assert batch.observations.dtype == batch.next_observations.dtype == batch.rewards.dtype == torch.float32
    assert batch.terminated.dtype == batch.truncated.dtype == torch.bool
    try:
        make_batch([])
        raise AssertionError('空批次未拒绝')
    except ValueError:
        pass
    online = nn.Linear(2, 3, bias=False)
    target = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        online.weight.copy_(torch.tensor([[0.1, 0.0], [0.5, 0.0], [0.2, 0.0]]))
    agent = Agent(online, target, torch.optim.SGD(online.parameters(), lr=0.01))
    agent.sync_target()
    before = online.weight.detach().clone()
    assert agent.act([1.0, 0.0], 0.0) == 1
    choices = {agent.act([1.0, 0.0], 1.0) for _ in range(100)}
    assert choices == {0, 1, 2}
    torch.testing.assert_close(online.weight, before)
    assert agent.updates == 0 and online.weight.grad is None
    stats = agent.update(batch)
    assert stats.updates == 1 and not torch.equal(before, online.weight)
    print('回放、批量转换、动作选择与一次更新连接通过：', stats)
    print('这是人工经验诊断；还没有持续游戏训练或策略效果结论。')


if __name__ == '__main__':
    try:
        checks()
    except NotImplementedError as error:
        print('实作尚未完成：', error)
        print('请完成replay.py和Agent.act；本次未记为检查通过。')
