import unittest

from examples.target_network_demo import (
    calculate_target_q,
    run_updates,
    update_online_weight,
)


class TargetNetworkDemoTest(unittest.TestCase):
    def test_non_terminal_target_uses_target_weight(self) -> None:
        target_q = calculate_target_q(
            reward=0.2,
            new_observation=1.0,
            target_weight=2.0,
            discount_factor=0.9,
            terminated=False,
        )

        self.assertAlmostEqual(target_q, 2.0)

    def test_terminal_target_ignores_target_weight(self) -> None:
        target_q = calculate_target_q(
            reward=0.2,
            new_observation=1.0,
            target_weight=99.0,
            discount_factor=0.9,
            terminated=True,
        )

        self.assertAlmostEqual(target_q, 0.2)

    def test_online_update_does_not_need_target_weight(self) -> None:
        new_online_weight = update_online_weight(
            online_weight=1.0,
            observation=1.0,
            error=0.1,
            learning_rate=0.2,
        )

        self.assertAlmostEqual(new_online_weight, 1.02)

    def test_target_stays_fixed_before_sync(self) -> None:
        records = run_updates(updates=2, sync_interval=3)

        self.assertEqual([record.target_after for record in records], [1.0, 1.0])
        self.assertAlmostEqual(records[0].target_q, records[1].target_q)

    def test_sync_copies_updated_online_weight(self) -> None:
        third_record = run_updates(updates=3, sync_interval=3)[-1]

        self.assertTrue(third_record.synced)
        self.assertAlmostEqual(
            third_record.target_after, third_record.online_after
        )

    def test_target_q_changes_only_after_sync(self) -> None:
        records = run_updates(updates=4, sync_interval=3)

        self.assertAlmostEqual(records[0].target_q, records[2].target_q)
        self.assertNotAlmostEqual(records[2].target_q, records[3].target_q)


if __name__ == "__main__":
    unittest.main()
