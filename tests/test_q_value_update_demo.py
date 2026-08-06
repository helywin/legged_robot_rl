import unittest

from examples.q_value_update_demo import update_q_value


class QValueUpdateDemoTest(unittest.TestCase):
    def test_learning_rate_point_two_moves_part_of_gap(self) -> None:
        new_q = update_q_value(old_q=0.3, reward=1.0, learning_rate=0.2)

        self.assertAlmostEqual(new_q, 0.44)

    def test_learning_rate_zero_keeps_old_q(self) -> None:
        new_q = update_q_value(old_q=0.3, reward=1.0, learning_rate=0.0)

        self.assertAlmostEqual(new_q, 0.3)

    def test_learning_rate_one_uses_new_reward(self) -> None:
        new_q = update_q_value(old_q=0.3, reward=1.0, learning_rate=1.0)

        self.assertAlmostEqual(new_q, 1.0)

    def test_invalid_learning_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            update_q_value(old_q=0.3, reward=1.0, learning_rate=-0.1)


if __name__ == "__main__":
    unittest.main()
