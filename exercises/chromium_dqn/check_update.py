"""教师诊断入口；执行学习者更新，不包含替代的update实现。"""
import math
import torch
from torch import nn
from network import QNetwork
from agent import Agent, Batch


def checks():
    torch.manual_seed(7)
    net = QNetwork()
    assert net(torch.zeros(2, 62)).shape == (2, 18)
    assert net(torch.zeros(1, 62)).shape == (1, 18)
    assert torch.isfinite(net(torch.ones(2, 62))).all()
    online = nn.Linear(2, 3, bias=False)
    target = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        online.weight.copy_(torch.tensor([[0.1, 0.2], [0.5, 0.6], [0.3, 0.2]]))
        target.weight.copy_(torch.tensor([[0.8, 0.7], [0.4, 0.6], [0.3, 0.5]]))
    agent = Agent(online, target, torch.optim.SGD(online.parameters(), lr=0.1), gamma=0.9)
    batch = Batch(torch.eye(2), torch.tensor([1, 2]), torch.tensor([0.2, 0.4]),
                  torch.eye(2), torch.tensor([False, True]), torch.tensor([True, False]))
    target_before = target.weight.detach().clone()
    stats = agent.update(batch)
    expected = torch.tensor([[0.1, 0.2], [0.542, 0.6], [0.3, 0.22]])
    assert math.isclose(stats.loss, 0.1082, abs_tol=1e-6), stats
    assert math.isclose(stats.mean_prediction, 0.35, abs_tol=1e-6)
    assert math.isclose(stats.mean_target, 0.66, abs_tol=1e-6)
    assert stats.updates == 1
    torch.testing.assert_close(online.weight, expected)
    torch.testing.assert_close(target.weight, target_before)
    assert target.weight.grad is None
    print('第一次更新:', stats, '\n在线权重:', online.weight.detach().tolist())
    second = agent.update(batch)
    assert second.updates == 2
    expected[1, 0], expected[2, 1] = 0.5798, 0.238
    torch.testing.assert_close(online.weight, expected)  # 检查是否清除了旧梯度。
    single = Batch(batch.observations[:1], batch.actions[:1], batch.rewards[:1],
                   batch.next_observations[:1], batch.terminated[:1], batch.truncated[:1])
    assert agent.update(single).updates == 3
    torch.testing.assert_close(target.weight, target_before)
    agent.sync_target()
    torch.testing.assert_close(target.weight, online.weight)
    assert target.weight.data_ptr() != online.weight.data_ptr()
    assert target.weight.grad is None
    print('参数更新、单样本批次、梯度清零和显式同步检查通过；尚非游戏训练。')


if __name__ == '__main__':
    try:
        checks()
    except NotImplementedError as error:
        print('本段实作尚未完成：', error)
        print('请完成network.py和agent.py；没有将本次运行记为训练通过。')
