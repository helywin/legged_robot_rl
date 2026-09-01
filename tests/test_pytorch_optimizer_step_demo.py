import unittest

import torch

from examples.pytorch_module_prediction import OneInputQModule
from examples.pytorch_optimizer_step_demo import make_one_optimizer_step


class PyTorchOptimizerStepDemoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = OneInputQModule(weight=1.0, bias=0.0)
        self.observation = torch.tensor(0.5)
        self.target = torch.tensor(1.0)
        self.result = make_one_optimizer_step(
            self.model,
            self.observation,
            self.target,
            learning_rate=0.1,
        )

    def test_gradient_is_computed_before_step(self) -> None:
        self.assertAlmostEqual(self.result.weight_gradient, -0.5)
        self.assertAlmostEqual(self.result.bias_gradient, -1.0)

    def test_step_changes_parameters_in_negative_gradient_direction(self) -> None:
        self.assertAlmostEqual(self.result.weight_after, 1.05)
        self.assertAlmostEqual(self.result.bias_after, 0.1)

    def test_prediction_moves_toward_target(self) -> None:
        self.assertAlmostEqual(self.result.prediction_before, 0.5)
        self.assertAlmostEqual(self.result.prediction_after, 0.625)

    def test_loss_decreases_after_step(self) -> None:
        self.assertAlmostEqual(self.result.loss_before, 0.25)
        self.assertAlmostEqual(self.result.loss_after, 0.140625)
        self.assertLess(self.result.loss_after, self.result.loss_before)


if __name__ == "__main__":
    unittest.main()
