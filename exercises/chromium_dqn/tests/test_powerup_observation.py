"""道具观察契约：类型、尺度、排序、容量、旧任务兼容。"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_task import fixture
from runtime import RawPowerUp
from task import encode_observation, encode_powerups, GameTask, OBSERVATION_SIZE
from network import QNetwork
import torch


class ObservationChecks(unittest.TestCase):
    def test_types_padding_and_old_prefix(self) -> None:
        base = fixture()
        for kind in range(6):
            item = RawPowerUp(1, kind, (2.0, 0.0, 25.0), (0.1, -0.2), 0.5)
            state = replace(base, powerups=(item,))
            vector = encode_observation(state)
            self.assertEqual(len(vector), 110)
            self.assertEqual(vector[:62], encode_observation(state, False))
            self.assertEqual(vector[62:68], [1.0, 0.1, 0.2, 0.1, -0.2, 0.5])
            self.assertEqual(vector[68:74], [float(i == kind) for i in range(6)])
            self.assertEqual(vector[74:], [0.0] * 36)
            self.assertEqual(state.powerups, (item,))
        self.assertEqual(encode_powerups(base), [0.0] * 48)
        self.assertEqual(QNetwork(OBSERVATION_SIZE)(torch.zeros(2, 110)).shape, (2, 18))

    def test_raw_distance_ties_capacity(self) -> None:
        # 水平8和竖直7的排序不可用缩放后的距离；最近两件同距按ID。
        positions = [(8.0, -3.0), (0.0, 4.0), (-1.0, -3.0), (1.0, -3.0), (9.0, -3.0)]
        ids = [4, 3, 2, 1, 5]
        items = tuple(RawPowerUp(i, 0, (x, y, 25.0), (0.0, -0.1), 1.0)
                      for i, (x, y) in zip(ids, positions))
        result = encode_powerups(replace(fixture(), powerups=items))
        self.assertEqual(len(result), 48)
        self.assertEqual([result[i] for i in (1, 13, 25, 37)], [0.05, -0.05, 0.0, 0.4])
        self.assertAlmostEqual(result[26], 7 / 15)
        with self.assertRaises(ValueError):
            GameTask(None, task_version='unknown')


if __name__ == '__main__':
    unittest.main()
