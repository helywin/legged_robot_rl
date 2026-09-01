import unittest

import torch

from examples.pytorch_selected_action_q_loss import (
    calculate_selected_action_loss,
)
from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


class PyTorchSelectedActionQLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TwoActionQModule()
        set_demo_parameters(self.model)
        self.observation = torch.tensor([0.2, -0.1])
        self.target = torch.tensor(0.5)

    def test_selected_q_uses_executed_action_not_best_action(self) -> None:
        q_values, selected_q, _loss = calculate_selected_action_loss(
            self.model,
            self.observation,
            executed_action=1,
            target=self.target,
        )

        self.assertEqual(q_values.argmax().item(), 0)
        self.assertAlmostEqual(selected_q.item(), -0.05)

    def test_loss_uses_only_selected_q(self) -> None:
        _q_values, _selected_q, loss = calculate_selected_action_loss(
            self.model,
            self.observation,
            executed_action=1,
            target=self.target,
        )

        self.assertAlmostEqual(loss.item(), 0.3025)

    def test_unselected_output_row_has_zero_gradient(self) -> None:
        calculate_selected_action_loss(
            self.model,
            self.observation,
            executed_action=1,
            target=self.target,
        )

        self.assertTrue(
            torch.equal(
                self.model.q_values.weight.grad[0], torch.tensor([0.0, 0.0])
            )
        )
        self.assertAlmostEqual(self.model.q_values.bias.grad[0].item(), 0.0)

    def test_selected_output_row_receives_gradient(self) -> None:
        calculate_selected_action_loss(
            self.model,
            self.observation,
            executed_action=1,
            target=self.target,
        )

        self.assertTrue(
            torch.allclose(
                self.model.q_values.weight.grad[1],
                torch.tensor([-0.22, 0.11]),
            )
        )
        self.assertAlmostEqual(
            self.model.q_values.bias.grad[1].item(), -1.1
        )


if __name__ == "__main__":
    unittest.main()
