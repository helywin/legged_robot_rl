"""五动作任务与原生按键的实际轨迹一致，旧版保持18动作。"""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Runtime
from task import (GameTask, FIVE_ACTION_TASK_VERSION, HIT_TASK_VERSION,
                  action_count_for, native_action_for)


class FiveActionChecks(unittest.TestCase):
    def test_invalid_and_legacy_actions(self) -> None:
        self.assertEqual(action_count_for(FIVE_ACTION_TASK_VERSION), 5)
        self.assertEqual(action_count_for(HIT_TASK_VERSION), 18)
        for action in range(18):
            self.assertEqual(native_action_for(action, HIT_TASK_VERSION), action)
        for action in (-1, 5, True, 1.0):
            with self.assertRaises(ValueError):
                native_action_for(action, FIVE_ACTION_TASK_VERSION)

    def test_real_trajectory_matches_held_fire(self) -> None:
        with Runtime(headless=True) as new_runtime, Runtime(headless=True) as old_runtime:
            new = GameTask(new_runtime, task_version=FIVE_ACTION_TASK_VERSION)
            old = GameTask(old_runtime, task_version=HIT_TASK_VERSION)
            for action, native_action in enumerate((9, 10, 11, 12, 13)):
                self.assertEqual(new.reset(31), old.reset(31))
                for _ in range(8):
                    self.assertEqual(new.step(action), old.step(native_action))


if __name__ == '__main__':
    unittest.main()
