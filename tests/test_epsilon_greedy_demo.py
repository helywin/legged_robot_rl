import random
import unittest

from examples.epsilon_greedy_demo import (
    EXPLOIT,
    EXPLORE,
    LEFT,
    Q_VALUES,
    RIGHT,
    choose_action,
    run_demo,
)


class EpsilonGreedyDemoTest(unittest.TestCase):
    def test_epsilon_zero_always_exploits_best_action(self) -> None:
        counts = run_demo(epsilon=0.0, decisions=100, seed=0)

        self.assertEqual(counts[EXPLORE], 0)
        self.assertEqual(counts[EXPLOIT], 100)
        self.assertEqual(counts[LEFT], 0)
        self.assertEqual(counts[RIGHT], 100)

    def test_epsilon_one_always_explores(self) -> None:
        counts = run_demo(epsilon=1.0, decisions=100, seed=0)

        self.assertEqual(counts[EXPLORE], 100)
        self.assertEqual(counts[EXPLOIT], 0)
        self.assertEqual(counts[LEFT] + counts[RIGHT], 100)

    def test_invalid_epsilon_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            choose_action(Q_VALUES, 1.1, random.Random(0))


if __name__ == "__main__":
    unittest.main()
