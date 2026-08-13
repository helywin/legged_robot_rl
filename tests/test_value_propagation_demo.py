import unittest

from examples.value_propagation_demo import capture_propagation


class ValuePropagationDemoTest(unittest.TestCase):
    def test_terminal_value_moves_one_position_earlier_each_episode(self) -> None:
        history = capture_propagation(episodes=4)

        self.assertEqual(history[0], [0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(history[1][3], 1.0)
        self.assertAlmostEqual(history[2][2], 0.89)
        self.assertAlmostEqual(history[3][1], 0.791)
        self.assertAlmostEqual(history[4][0], 0.7019)

    def test_invalid_episode_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            capture_propagation(episodes=0)


if __name__ == "__main__":
    unittest.main()
