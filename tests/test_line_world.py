import unittest

from examples.line_world import LEFT, RIGHT, LineWorldEnv


class LineWorldEnvTest(unittest.TestCase):
    def test_right_actions_reach_goal_in_four_steps(self) -> None:
        env = LineWorldEnv()
        observation = env.reset()
        total_reward = 0.0

        self.assertEqual(observation, 0)
        for _ in range(4):
            observation, reward, done = env.step(RIGHT)
            total_reward += reward

        self.assertEqual(observation, 4)
        self.assertTrue(done)
        self.assertAlmostEqual(total_reward, 0.97)

    def test_left_action_does_not_leave_world(self) -> None:
        env = LineWorldEnv()
        env.reset()

        observation, reward, done = env.step(LEFT)

        self.assertEqual(observation, 0)
        self.assertEqual(reward, -0.01)
        self.assertFalse(done)

    def test_episode_ends_when_step_limit_is_reached(self) -> None:
        env = LineWorldEnv(max_steps=2)
        env.reset()
        env.step(LEFT)

        observation, reward, done = env.step(LEFT)

        self.assertEqual(observation, 0)
        self.assertEqual(reward, -0.01)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
