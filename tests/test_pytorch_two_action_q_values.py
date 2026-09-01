import unittest
from typing import get_type_hints

import torch
from torch import nn

from examples.pytorch_two_action_q_values import (
    TwoActionQModule,
    set_demo_parameters,
)


class PyTorchTwoActionQValuesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TwoActionQModule()
        set_demo_parameters(self.model)

    def test_module_contains_two_by_two_linear_layer(self) -> None:
        self.assertIsInstance(self.model.q_values, nn.Linear)
        self.assertEqual(tuple(self.model.q_values.weight.shape), (2, 2))
        self.assertEqual(tuple(self.model.q_values.bias.shape), (2,))

    def test_call_signature_exposes_tensor_type(self) -> None:
        hints = get_type_hints(TwoActionQModule.__call__)

        self.assertIs(hints["observation"], torch.Tensor)
        self.assertIs(hints["return"], torch.Tensor)

    def test_prediction_returns_one_q_value_per_action(self) -> None:
        q_values = self.model(torch.tensor([0.2, -0.1]))

        self.assertEqual(tuple(q_values.shape), (2,))
        self.assertTrue(
            torch.allclose(q_values, torch.tensor([0.1, -0.05]))
        )

    def test_different_observation_can_change_best_action(self) -> None:
        first = self.model(torch.tensor([0.2, -0.1]))
        second = self.model(torch.tensor([-1.0, 0.0]))

        self.assertEqual(first.argmax().item(), 0)
        self.assertEqual(second.argmax().item(), 1)

    def test_prediction_does_not_change_parameters(self) -> None:
        before = [parameter.detach().clone() for parameter in self.model.parameters()]

        self.model(torch.tensor([0.2, -0.1]))

        self.assertTrue(
            all(
                torch.equal(old, current)
                for old, current in zip(before, self.model.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
