import unittest

from examples.frozen_lake import DOWN, LEFT, RIGHT, FrozenLakeEnv


class FrozenLakeEnvTest(unittest.TestCase):
    def test_reset_returns_initial_observation_and_info(self) -> None:
        env = FrozenLakeEnv()

        observation, info = env.reset()

        self.assertEqual(observation, 0)
        self.assertEqual(info["position"], (0, 0))
        self.assertEqual(info["tile"], "S")
        self.assertEqual(env.observation_count, 16)
        self.assertEqual(len(env.action_space), 4)

    def test_moving_outside_map_stays_in_same_cell(self) -> None:
        env = FrozenLakeEnv()
        env.reset()

        observation, reward, terminated, truncated, _ = env.step(LEFT)

        self.assertEqual(observation, 0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_hole_terminates_episode_without_reward(self) -> None:
        env = FrozenLakeEnv()
        env.reset()
        env.step(DOWN)

        observation, reward, terminated, truncated, info = env.step(RIGHT)

        self.assertEqual(observation, 5)
        self.assertEqual(info["tile"], "H")
        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_goal_terminates_episode_with_reward(self) -> None:
        env = FrozenLakeEnv()
        env.reset()
        for action in (DOWN, DOWN, RIGHT, DOWN, RIGHT):
            _, reward, terminated, truncated, _ = env.step(action)
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)

        observation, reward, terminated, truncated, info = env.step(RIGHT)

        self.assertEqual(observation, 15)
        self.assertEqual(info["tile"], "G")
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_step_limit_truncates_without_termination(self) -> None:
        env = FrozenLakeEnv(max_steps=1)
        env.reset()

        _, _, terminated, truncated, _ = env.step(LEFT)

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_termination_takes_priority_at_step_limit(self) -> None:
        env = FrozenLakeEnv(max_steps=2)
        env.reset()
        env.step(DOWN)

        _, _, terminated, truncated, _ = env.step(RIGHT)

        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_rejects_unknown_action(self) -> None:
        env = FrozenLakeEnv()
        env.reset()

        with self.assertRaises(ValueError):
            env.step("jump")

    def test_rejects_step_after_episode_ends(self) -> None:
        env = FrozenLakeEnv(max_steps=1)
        env.reset()
        env.step(LEFT)

        with self.assertRaises(RuntimeError):
            env.step(LEFT)

    def test_render_marks_agent_position(self) -> None:
        env = FrozenLakeEnv()
        env.reset()

        rendered = env.render()

        self.assertTrue(rendered.startswith("[A][F][F][F]"))
        self.assertIn("[G]", rendered)


if __name__ == "__main__":
    unittest.main()
