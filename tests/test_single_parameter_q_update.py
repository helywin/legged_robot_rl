import unittest

from examples.single_parameter_q_update import (
    calculate_prediction_error,
    predict_q,
    run_updates,
    update_weight,
)


class SingleParameterQUpdateTest(unittest.TestCase):
    def test_positive_error_means_prediction_is_too_low(self) -> None:
        self.assertAlmostEqual(
            calculate_prediction_error(target_q=1.0, predicted_q=0.7), 0.3
        )

    def test_negative_error_means_prediction_is_too_high(self) -> None:
        self.assertAlmostEqual(
            calculate_prediction_error(target_q=1.0, predicted_q=1.3), -0.3
        )

    def test_one_update_moves_prediction_toward_target(self) -> None:
        observation = 0.5
        old_weight = 1.0
        old_prediction = predict_q(observation, old_weight)
        error = calculate_prediction_error(1.0, old_prediction)

        new_weight = update_weight(old_weight, observation, error, 0.2)
        new_prediction = predict_q(observation, new_weight)

        self.assertLess(abs(1.0 - new_prediction), abs(1.0 - old_prediction))

    def test_repeated_updates_keep_reducing_absolute_error(self) -> None:
        records = run_updates(0.5, 1.0, 0.2, 1.0, 6)
        absolute_errors = [abs(error) for _, _, error in records]

        self.assertTrue(
            all(
                later < earlier
                for earlier, later in zip(absolute_errors, absolute_errors[1:])
            )
        )

    def test_invalid_learning_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            update_weight(1.0, 0.5, 0.5, 1.1)


if __name__ == "__main__":
    unittest.main()
