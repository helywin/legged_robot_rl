import unittest

import torch

from examples.pytorch_dqn_target_q import calculate_dqn_target
from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


class PyTorchDqnTargetQTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target_network = TwoActionQModule()
        set_demo_parameters(self.target_network)
        self.next_observation = torch.tensor([0.4, 0.2])
        self.reward = torch.tensor(0.2)

    def test_non_terminal_target_uses_best_next_q(self) -> None:
        target = calculate_dqn_target(
            self.target_network,
            self.next_observation,
            self.reward,
            discount_factor=0.9,
            terminated=False,
        )

        self.assertAlmostEqual(target.item(), 1.01)

    def test_terminal_target_uses_reward_only(self) -> None:
        target = calculate_dqn_target(
            self.target_network,
            self.next_observation,
            self.reward,
            discount_factor=0.9,
            terminated=True,
        )

        self.assertAlmostEqual(target.item(), 0.2)

    def test_terminal_target_does_not_call_target_network(self) -> None:
        calls = 0

        def remember_call(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            nonlocal calls
            calls += 1

        hook = self.target_network.register_forward_hook(remember_call)
        try:
            calculate_dqn_target(
                self.target_network,
                self.next_observation,
                self.reward,
                discount_factor=0.9,
                terminated=True,
            )
        finally:
            hook.remove()

        self.assertEqual(calls, 0)

    def test_target_is_detached_and_target_network_has_no_gradients(self) -> None:
        target = calculate_dqn_target(
            self.target_network,
            self.next_observation,
            self.reward,
            discount_factor=0.9,
            terminated=False,
        )

        self.assertFalse(target.requires_grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in self.target_network.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
