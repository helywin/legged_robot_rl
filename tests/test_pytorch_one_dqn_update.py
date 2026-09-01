import unittest

import torch

from examples.pytorch_one_dqn_update import (
    DqnUpdateResult,
    update_from_one_transition,
)
from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


class PyTorchOneDqnUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.online_network = TwoActionQModule()
        self.target_network = TwoActionQModule()
        set_demo_parameters(self.online_network)
        set_demo_parameters(self.target_network)
        self.optimizer = torch.optim.SGD(
            self.online_network.parameters(), lr=0.1
        )
        self.observation = torch.tensor([0.2, -0.1])
        self.next_observation = torch.tensor([0.4, 0.2])
        self.reward = torch.tensor(0.2)

    def run_update(self) -> DqnUpdateResult:
        return update_from_one_transition(
            self.online_network,
            self.target_network,
            self.optimizer,
            self.observation,
            executed_action=1,
            reward=self.reward,
            next_observation=self.next_observation,
            discount_factor=0.9,
            terminated=False,
        )

    def test_update_uses_expected_prediction_target_and_loss(self) -> None:
        result = self.run_update()

        self.assertEqual(result.q_values_before, [0.1, -0.05])
        self.assertAlmostEqual(result.selected_q_before, -0.05)
        self.assertAlmostEqual(result.target_q, 1.01)
        self.assertAlmostEqual(result.loss, 1.1236, places=6)

    def test_selected_q_moves_toward_target(self) -> None:
        result = self.run_update()

        self.assertAlmostEqual(result.selected_q_after, 0.1726, delta=1e-6)
        self.assertLess(
            abs(result.selected_q_after - result.target_q),
            abs(result.selected_q_before - result.target_q),
        )

    def test_only_selected_online_output_row_changes(self) -> None:
        weight_before = self.online_network.q_values.weight.detach().clone()
        bias_before = self.online_network.q_values.bias.detach().clone()

        self.run_update()

        self.assertTrue(
            torch.equal(
                weight_before[0], self.online_network.q_values.weight[0]
            )
        )
        self.assertEqual(
            bias_before[0].item(), self.online_network.q_values.bias[0].item()
        )
        self.assertFalse(
            torch.equal(
                weight_before[1], self.online_network.q_values.weight[1]
            )
        )

    def test_target_network_stays_unchanged_without_gradients(self) -> None:
        before = [
            parameter.detach().clone()
            for parameter in self.target_network.parameters()
        ]

        self.run_update()

        self.assertTrue(
            all(
                torch.equal(old, current)
                for old, current in zip(before, self.target_network.parameters())
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in self.target_network.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
