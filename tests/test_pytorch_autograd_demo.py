import unittest

import torch

from examples.pytorch_autograd_demo import calculate_loss_and_gradients
from examples.pytorch_module_prediction import OneInputQModule


class PyTorchAutogradDemoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = OneInputQModule(weight=1.0, bias=0.0)
        self.observation = torch.tensor(0.5)
        self.target = torch.tensor(1.0)

    def test_squared_loss_has_expected_value(self) -> None:
        prediction, loss = calculate_loss_and_gradients(
            self.model, self.observation, self.target
        )

        self.assertAlmostEqual(prediction.item(), 0.5)
        self.assertAlmostEqual(loss.item(), 0.25)

    def test_backward_populates_parameter_gradients(self) -> None:
        calculate_loss_and_gradients(self.model, self.observation, self.target)

        self.assertAlmostEqual(self.model.weight.grad.item(), -0.5)
        self.assertAlmostEqual(self.model.bias.grad.item(), -1.0)

    def test_backward_does_not_update_parameters(self) -> None:
        before = [parameter.detach().clone() for parameter in self.model.parameters()]

        calculate_loss_and_gradients(self.model, self.observation, self.target)

        self.assertTrue(
            all(
                torch.equal(old, current)
                for old, current in zip(before, self.model.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
