import unittest

from examples.env_interface_demo import (
    ADVANCE,
    RETREAT,
    RoomEnv,
    make_fixed_policy,
    run_episode,
)
from examples.line_world import RIGHT, LineWorldEnv


class RunEpisodeTest(unittest.TestCase):
    def test_same_loop_runs_line_world(self) -> None:
        total_reward, steps = run_episode(LineWorldEnv(), make_fixed_policy(RIGHT))

        self.assertAlmostEqual(total_reward, 0.97)
        self.assertEqual(steps, 4)

    def test_same_loop_runs_room_env(self) -> None:
        total_reward, steps = run_episode(RoomEnv(), make_fixed_policy(ADVANCE))

        self.assertAlmostEqual(total_reward, 0.98)
        self.assertEqual(steps, 2)

    def test_room_env_times_out_when_never_reaching_goal(self) -> None:
        total_reward, steps = run_episode(RoomEnv(), make_fixed_policy(RETREAT))

        self.assertAlmostEqual(total_reward, -0.12)
        self.assertEqual(steps, 6)


class RoomEnvTest(unittest.TestCase):
    def test_reset_returns_first_room_name(self) -> None:
        env = RoomEnv()

        self.assertEqual(env.reset(), "门口")

    def test_rejects_unknown_action(self) -> None:
        env = RoomEnv()
        env.reset()

        with self.assertRaises(ValueError):
            env.step("jump")

    def test_rejects_step_after_done(self) -> None:
        env = RoomEnv(max_steps=1)
        env.reset()
        env.step(RETREAT)

        with self.assertRaises(RuntimeError):
            env.step(RETREAT)


if __name__ == "__main__":
    unittest.main()
