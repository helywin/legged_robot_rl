import unittest

from examples.python_function_model import predict_q


class PythonFunctionModelTest(unittest.TestCase):
    def test_prediction_multiplies_then_adds_bias(self) -> None:
        self.assertAlmostEqual(
            predict_q(observation=-0.02, weight=3.0, bias=0.1),
            0.04,
        )

    def test_same_function_can_use_different_inputs(self) -> None:
        first = predict_q(observation=0.2, weight=2.0, bias=0.1)
        second = predict_q(observation=0.5, weight=2.0, bias=0.1)

        self.assertAlmostEqual(first, 0.5)
        self.assertAlmostEqual(second, 1.1)

    def test_arguments_are_not_modified(self) -> None:
        observation = 0.2
        weight = 2.0
        bias = 0.1

        predict_q(observation, weight, bias)

        self.assertEqual(observation, 0.2)
        self.assertEqual(weight, 2.0)
        self.assertEqual(bias, 0.1)


if __name__ == "__main__":
    unittest.main()
