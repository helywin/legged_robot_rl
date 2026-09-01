import unittest

from examples.pytorch_zero_grad_demo import (
    prepare_first_update,
    update_on_second_target,
)


class PyTorchZeroGradDemoTest(unittest.TestCase):
    def test_first_update_leaves_old_gradient(self) -> None:
        model, _optimizer = prepare_first_update()

        self.assertAlmostEqual(model.weight.grad.item(), -0.5)
        self.assertAlmostEqual(model.bias.grad.item(), -1.0)

    def test_without_clear_backward_accumulates_old_gradient(self) -> None:
        result = update_on_second_target(clear_old_gradient=False)

        self.assertAlmostEqual(result.weight_gradient, -0.275)
        self.assertAlmostEqual(result.bias_gradient, -0.55)

    def test_without_clear_prediction_moves_away_from_second_target(self) -> None:
        result = update_on_second_target(clear_old_gradient=False)

        self.assertFalse(result.moved_toward_target)
        self.assertAlmostEqual(result.prediction_after, 0.69375)

    def test_clear_uses_only_second_gradient_and_moves_toward_target(self) -> None:
        result = update_on_second_target(clear_old_gradient=True)

        self.assertAlmostEqual(result.weight_gradient, 0.225)
        self.assertAlmostEqual(result.bias_gradient, 0.45)
        self.assertTrue(result.moved_toward_target)
        self.assertAlmostEqual(result.prediction_after, 0.56875)
        self.assertLess(result.loss_after, result.loss_before)


if __name__ == "__main__":
    unittest.main()
