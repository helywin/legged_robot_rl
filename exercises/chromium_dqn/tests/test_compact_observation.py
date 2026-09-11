"""压缩实验：被删运动字段不能影响输入，奖励与历史模型契约不变。"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_task import fixture
from runtime import RawBullet, RawPowerUp, RawEvents
from task import COMPACT_TASK_VERSION, encode_task_observation, compute_reward, observation_size_for


class CompactChecks(unittest.TestCase):
    def test_removed_fields_and_retained_information(self) -> None:
        state = replace(fixture(), events=RawEvents(),
                        enemy_bullets=(RawBullet(1, 0, (2., 1., 25.), (0.1, -0.2, 0.), (0.1, 0.1), 1.),),
                        powerups=(RawPowerUp(1, 3, (2., 0., 25.), (0.1, -0.2), 0.5),))
        changed = replace(state, player=replace(state.player, keyboard_motion=(99., -99.)),
                          enemy_bullets=(replace(state.enemy_bullets[0], velocity_per_tick=(-9., 9., 0.)),),
                          powerups=(replace(state.powerups[0], next_displacement=(-8., 8.)),))
        compact = encode_task_observation(state, COMPACT_TASK_VERSION)
        self.assertEqual(len(compact), 84)
        self.assertEqual(compact, encode_task_observation(changed, COMPACT_TASK_VERSION))
        self.assertNotEqual(encode_task_observation(state, 'task-v3-events'),
                            encode_task_observation(changed, 'task-v3-events'))
        self.assertEqual(compact[44:54], [1., .1, .2, .5, 0., 0., 0., 1., 0., 0.])
        self.assertEqual(compact[54:], [0.] * 30)
        for changed in (
            replace(state, player=replace(state.player, shields=0.)),
            replace(state, enemy_bullets=()),
            replace(state, powerups=(replace(state.powerups[0], type=4),)),
            replace(state, powerups=(replace(state.powerups[0], position=(3., 0., 25.)),)),
        ):
            self.assertNotEqual(compact, encode_task_observation(changed, COMPACT_TASK_VERSION))

    def test_version_dimensions_and_identical_reward(self) -> None:
        before = replace(fixture(), events=RawEvents())
        after = replace(before, events=RawEvents(enemies_destroyed=2, pickups=1, lives_lost=1))
        for version, size in [('task-v1', 62), ('task-v2-powerups', 110),
                              ('task-v3-events', 110), (COMPACT_TASK_VERSION, 84)]:
            self.assertEqual(observation_size_for(version), size)
            self.assertEqual(len(encode_task_observation(before, version)), size)
        self.assertAlmostEqual(compute_reward(before, after, COMPACT_TASK_VERSION), 15.2)
        self.assertEqual(compute_reward(before, after, COMPACT_TASK_VERSION),
                         compute_reward(before, after, 'task-v3-events'))


if __name__ == '__main__':
    unittest.main()
