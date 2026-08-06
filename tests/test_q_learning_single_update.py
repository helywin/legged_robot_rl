import unittest

from examples.q_learning_single_update import calculate_target, update_q_value


class QLearningSingleUpdateTest(unittest.TestCase):
    def test_non_terminal_update_uses_best_next_q(self) -> None:
        target = calculate_target(
            reward=-0.01,
            best_next_q=0.8,
            discount_factor=0.9,
            done=False,
        )
        new_q = update_q_value(old_q=0.3, target=target, learning_rate=0.2)

        self.assertAlmostEqual(target, 0.71)
        self.assertAlmostEqual(new_q, 0.382)

    def test_terminal_target_uses_only_reward(self) -> None:
        target = calculate_target(
            reward=1.0,
            best_next_q=99.0,
            discount_factor=0.9,
            done=True,
        )

        self.assertAlmostEqual(target, 1.0)

    def test_invalid_discount_factor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_target(0.0, 0.0, 1.1, False)

    def test_invalid_learning_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            update_q_value(old_q=0.0, target=1.0, learning_rate=1.1)


if __name__ == "__main__":
    unittest.main()
