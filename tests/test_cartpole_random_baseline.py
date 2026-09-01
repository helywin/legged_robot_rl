import unittest

import gymnasium as gym

from examples.cartpole_random_baseline import (
    run_random_baseline,
    run_random_episode,
)


class CartPoleRandomEpisodeTest(unittest.TestCase):
    def test_each_step_contributes_one_reward(self) -> None:
        env = gym.make("CartPole-v1")
        try:
            result = run_random_episode(env, seed=20260901)
        finally:
            env.close()

        self.assertEqual(result.total_reward, float(result.steps))
        self.assertTrue(result.terminated or result.truncated)
        self.assertFalse(result.terminated and result.truncated)


class CartPoleRandomBaselineTest(unittest.TestCase):
    def test_same_seeds_reproduce_same_summary(self) -> None:
        first = run_random_baseline(episodes=5, base_seed=20260901)
        second = run_random_baseline(episodes=5, base_seed=20260901)

        self.assertEqual(first, second)

    def test_every_episode_has_one_end_reason(self) -> None:
        summary = run_random_baseline(episodes=5, base_seed=20260901)

        self.assertEqual(
            summary.terminated_episodes + summary.truncated_episodes,
            summary.episodes,
        )

    def test_rejects_non_positive_episode_count(self) -> None:
        with self.assertRaises(ValueError):
            run_random_baseline(episodes=0)


if __name__ == "__main__":
    unittest.main()
