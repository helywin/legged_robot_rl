import unittest

import torch

from examples.pytorch_module_prediction import OneInputQModule
from examples.pytorch_training_loop_demo import train_fixed_sample


class PyTorchTrainingLoopDemoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = OneInputQModule(weight=1.0, bias=0.0)
        self.observation = torch.tensor(0.5)
        self.target = torch.tensor(1.0)

    def test_rejects_non_positive_step_count(self) -> None:
        with self.assertRaises(ValueError):
            train_fixed_sample(
                self.model,
                self.observation,
                self.target,
                learning_rate=0.1,
                steps=0,
            )

    def test_returns_one_loss_per_training_step(self) -> None:
        losses = train_fixed_sample(
            self.model,
            self.observation,
            self.target,
            learning_rate=0.1,
            steps=8,
        )

        self.assertEqual(len(losses), 8)
        self.assertAlmostEqual(losses[0], 0.25)

    def test_loss_keeps_decreasing(self) -> None:
        losses = train_fixed_sample(
            self.model,
            self.observation,
            self.target,
            learning_rate=0.1,
            steps=8,
        )

        self.assertTrue(all(a > b for a, b in zip(losses, losses[1:])))

    def test_final_prediction_moves_close_to_target(self) -> None:
        train_fixed_sample(
            self.model,
            self.observation,
            self.target,
            learning_rate=0.1,
            steps=8,
        )

        with torch.no_grad():
            prediction = self.model(self.observation)
        self.assertAlmostEqual(
            prediction.item(), 0.94994366, delta=1e-6
        )
        self.assertLess(abs(prediction.item() - self.target.item()), 0.051)


if __name__ == "__main__":
    unittest.main()
