"""36维槽位裁剪、保留信息和v5奖励一致性。"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_task import fixture
from runtime import RawPowerUp, RawEvents
from task import encode_task_observation, observation_size_for, compute_reward, SMALL_TASK_VERSION, HIT_TASK_VERSION, FIVE_ACTION_TASK_VERSION
from train import TrainConfig


class SmallObservationChecks(unittest.TestCase):
    def test_exact_slots_and_padding(self) -> None:
        state = replace(fixture(), powerups=(
            RawPowerUp(2, 0, (8., 0., 25.), (0., 0.), 1.),
            RawPowerUp(1, 3, (1., -3., 25.), (0., 0.), .5),
        ))
        full = encode_task_observation(state, 'task-v5-shield-damage')
        small = encode_task_observation(state, SMALL_TASK_VERSION)
        self.assertEqual(len(small), 36)
        self.assertEqual(small[:8], full[:8])
        self.assertEqual(small[8:14], full[8:14])
        self.assertEqual(small[14:26], full[20:32])
        self.assertEqual(small[26:], [1., .05, 0., .5, 0., 0., 0., 1., 0., 0.])
        empty = replace(state, enemies=(), enemy_bullets=(), powerups=())
        self.assertEqual(encode_task_observation(empty, SMALL_TASK_VERSION)[8:], [0.] * 28)
        self.assertEqual(observation_size_for(SMALL_TASK_VERSION), 36)
        self.assertEqual(TrainConfig().task_version, FIVE_ACTION_TASK_VERSION)

    def test_reward_same_as_current_v5(self) -> None:
        before = replace(fixture(), events=RawEvents())
        after = replace(before, events=RawEvents(enemies_destroyed=2, pickups=1,
                                                lives_lost=1, shield_damage=40.))
        self.assertEqual(compute_reward(before, after, SMALL_TASK_VERSION),
                         compute_reward(before, after, 'task-v5-shield-damage'))


if __name__ == '__main__':
    unittest.main()
