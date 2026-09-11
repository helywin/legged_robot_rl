"""编译路径与学习版逐次对照，覆盖终止/截断、目标同步与不支持的优化器。"""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Agent, Batch
from fast_update import ScriptedSGDAgent
from network import QNetwork


class FastUpdateChecks(unittest.TestCase):
    def test_exact_updates_targets_and_shared_parameters(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(17)
        online, fast_online = QNetwork(84), QNetwork(84)
        fast_online.load_state_dict(online.state_dict())
        eager = Agent(online, QNetwork(84), torch.optim.SGD(online.parameters(), lr=.001))
        fast = ScriptedSGDAgent(fast_online, QNetwork(84),
                               torch.optim.SGD(fast_online.parameters(), lr=.001))
        eager.sync_target()
        fast.sync_target()
        target_before = {name: value.clone() for name, value in fast.target.state_dict().items()}
        for i in range(200):
            batch = Batch(torch.randn(32, 84), torch.randint(18, (32,)),
                          torch.randn(32), torch.randn(32, 84),
                          torch.rand(32) < .2, torch.rand(32) < .2)
            self.assertEqual(eager.update(batch), fast.update(batch))
            for left, right in zip(online.parameters(), fast_online.parameters()):
                self.assertTrue(torch.equal(left, right))
            for name, value in fast.target.state_dict().items():
                self.assertTrue(torch.equal(value, target_before[name]))
            if (i + 1) % 20 == 0:
                eager.sync_target()
                fast.sync_target()
                target_before = {name: value.clone() for name, value in fast.target.state_dict().items()}
        self.assertTrue(all(p.grad is None for p in fast.target.parameters()))

    def test_reject_non_plain_sgd(self) -> None:
        for optimizer_type, kwargs in [(torch.optim.Adam, {}), (torch.optim.SGD, {'momentum': .9})]:
            online = QNetwork(84)
            with self.assertRaises(ValueError):
                ScriptedSGDAgent(online, QNetwork(84), optimizer_type(online.parameters(), lr=.001, **kwargs))


if __name__ == '__main__':
    unittest.main()
