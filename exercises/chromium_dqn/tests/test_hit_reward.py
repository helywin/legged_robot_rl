"""命中反馈分段计算，非射击清场不应被当作攻击收益。"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_task import fixture
from runtime import RawEvents
from task import compute_reward, encode_task_observation, HIT_TASK_VERSION


class HitRewardChecks(unittest.TestCase):
    def test_each_partial_hit_rewards_before_kill(self) -> None:
        before = replace(fixture(), events=RawEvents())
        rewards: list[float] = []
        for fraction, kills in [(.2, 0), (.5, 0), (1., 1)]:
            after = replace(before, events=RawEvents(projectile_damage_fraction=fraction,
                                                    projectile_damage=100*fraction,
                                                    projectile_kills=kills))
            rewards.append(compute_reward(before, after, HIT_TASK_VERSION))
            before = after
        for actual, expected in zip(rewards, [.2, .3, 2.5]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(rewards), 3.)
        self.assertEqual(len(encode_task_observation(before, HIT_TASK_VERSION)), 36)

    def test_non_projectile_clear_does_not_offset_death(self) -> None:
        before = replace(fixture(), events=RawEvents())
        after = replace(before, events=RawEvents(enemies_destroyed=2, lives_lost=1))
        self.assertEqual(compute_reward(before, after, HIT_TASK_VERSION), -5.)
        self.assertEqual(compute_reward(before, after, 'task-v6-36'), 15.)
        after = replace(before, events=RawEvents(shield_damage=40., pickups=1,
                                                projectile_damage_fraction=.25))
        self.assertAlmostEqual(compute_reward(before, after, HIT_TASK_VERSION), .05)


if __name__ == '__main__':
    unittest.main()
