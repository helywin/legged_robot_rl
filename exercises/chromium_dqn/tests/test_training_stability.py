"""大TD误差、编译等价性、隐藏层失活检测及经验形状检查。"""
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Agent, Batch
from fast_update import ScriptedSGDAgent
from network import QNetwork
from network_health import inspect_network
from tensor_replay import TensorReplay
from replay import Transition


class StabilityChecks(unittest.TestCase):
    def test_huber_limits_large_error_gradient(self) -> None:
        model, target = nn.Linear(1, 1), nn.Linear(1, 1)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        agent = Agent(model, target, torch.optim.SGD(model.parameters(), lr=.001), loss_kind='huber')
        batch = Batch(torch.ones(1, 1), torch.zeros(1, dtype=torch.int64), torch.tensor([1e6]),
                      torch.ones(1, 1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
        agent.update(batch)
        self.assertAlmostEqual(model.weight.item(), .001)
        self.assertAlmostEqual(model.bias.item(), .001)
        self.assertIsNone(target.weight.grad)

    def test_huber_eager_and_scripted_match(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(7)
        online, compiled = QNetwork(36), QNetwork(36)
        compiled.load_state_dict(online.state_dict())
        a = Agent(online, QNetwork(36), torch.optim.SGD(online.parameters(), lr=.001), loss_kind='huber')
        b = ScriptedSGDAgent(compiled, QNetwork(36), torch.optim.SGD(compiled.parameters(), lr=.001), loss_kind='huber')
        a.sync_target()
        b.sync_target()
        for i in range(100):
            batch = Batch(torch.randn(32, 36), torch.randint(18, (32,)), torch.randn(32) * 100,
                          torch.randn(32, 36), torch.rand(32) < .2, torch.rand(32) < .2)
            self.assertEqual(a.update(batch), b.update(batch))
            self.assertTrue(all(torch.equal(p, q) for p, q in zip(online.parameters(), compiled.parameters())))
            if i % 10 == 0:
                a.sync_target()
                b.sync_target()

    def test_dead_hidden_layer_detected_without_mutation(self) -> None:
        network = QNetwork(36)
        with torch.no_grad():
            network.layers[0].weight.zero_()
            network.layers[0].bias.fill_(-1)
            network.layers[2].bias.fill_(-1)
        before = {name: value.clone() for name, value in network.state_dict().items()}
        result = inspect_network(network, torch.randn(20, 36))
        self.assertTrue(result['collapsed'])
        self.assertEqual(result['q_state_span'], 0)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in network.state_dict().items()))

    def test_replay_rejects_broadcast_observations(self) -> None:
        replay = TensorReplay(10, 36)
        with self.assertRaises(ValueError):
            replay.add(Transition((1.,), 0, 0., (2.,), False, False))
        self.assertEqual(len(replay), 0)


if __name__ == '__main__':
    unittest.main()
