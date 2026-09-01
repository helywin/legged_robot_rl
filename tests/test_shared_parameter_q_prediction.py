import unittest

from examples.shared_parameter_q_prediction import (
    DEFAULT_PARAMETERS,
    LEFT,
    RIGHT,
    choose_best_action,
    predict_q_values,
)


class SharedParameterQPredictionTest(unittest.TestCase):
    def test_prediction_returns_one_q_value_per_action(self) -> None:
        q_values = predict_q_values((0.2, 0.1), DEFAULT_PARAMETERS)

        self.assertEqual(len(q_values), 2)
        self.assertAlmostEqual(q_values[0], 0.10)
        self.assertAlmostEqual(q_values[1], 0.09)

    def test_different_observation_can_change_best_action(self) -> None:
        first = choose_best_action(
            predict_q_values((0.2, 0.1), DEFAULT_PARAMETERS)
        )
        second = choose_best_action(
            predict_q_values((0.8, 0.1), DEFAULT_PARAMETERS)
        )

        self.assertEqual(first, LEFT)
        self.assertEqual(second, RIGHT)

    def test_prediction_does_not_modify_parameters(self) -> None:
        parameters_before = dict(DEFAULT_PARAMETERS)

        predict_q_values((0.8, 0.9), DEFAULT_PARAMETERS)

        self.assertEqual(DEFAULT_PARAMETERS, parameters_before)

    def test_best_action_rejects_wrong_number_of_q_values(self) -> None:
        with self.assertRaises(ValueError):
            choose_best_action([0.1])


if __name__ == "__main__":
    unittest.main()
