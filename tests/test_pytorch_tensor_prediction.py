import unittest

import torch

from examples.pytorch_tensor_prediction import predict_q


class PyTorchTensorPredictionTest(unittest.TestCase):
    def test_prediction_returns_scalar_tensor(self) -> None:
        prediction = predict_q(
            observation=torch.tensor(-0.02),
            weight=torch.tensor(3.0),
            bias=torch.tensor(0.1),
        )

        self.assertIsInstance(prediction, torch.Tensor)
        self.assertEqual(prediction.ndim, 0)
        self.assertAlmostEqual(prediction.item(), 0.04)

    def test_different_tensor_input_changes_prediction(self) -> None:
        weight = torch.tensor(3.0)
        bias = torch.tensor(0.1)

        first = predict_q(torch.tensor(-0.02), weight, bias)
        second = predict_q(torch.tensor(0.05), weight, bias)

        self.assertAlmostEqual(first.item(), 0.04)
        self.assertAlmostEqual(second.item(), 0.25)

    def test_prediction_does_not_modify_input_tensors(self) -> None:
        observation = torch.tensor(-0.02)
        weight = torch.tensor(3.0)
        bias = torch.tensor(0.1)
        before = (observation.clone(), weight.clone(), bias.clone())

        predict_q(observation, weight, bias)

        self.assertTrue(torch.equal(observation, before[0]))
        self.assertTrue(torch.equal(weight, before[1]))
        self.assertTrue(torch.equal(bias, before[2]))


if __name__ == "__main__":
    unittest.main()
