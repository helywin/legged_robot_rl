import random
import unittest

from examples.train_line_world_q_learning import (
    ACTIONS,
    choose_action_index,
    create_q_table,
    evaluate_q_table,
    train_q_table,
)


class TrainLineWorldQLearningTest(unittest.TestCase):
    def test_q_table_has_one_row_per_state_and_one_column_per_action(self) -> None:
        q_table = create_q_table(state_count=5, action_count=2)

        self.assertEqual(len(q_table), 5)
        self.assertTrue(all(len(row) == 2 for row in q_table))
        self.assertTrue(all(value == 0.0 for row in q_table for value in row))

    def test_training_learns_right_as_best_action(self) -> None:
        q_table, _ = train_q_table(episodes=500, seed=0)

        for observation in range(4):
            best_action_index = max(
                range(len(ACTIONS)), key=q_table[observation].__getitem__
            )
            self.assertEqual(ACTIONS[best_action_index], "right")

    def test_evaluation_reaches_goal_without_changing_q_table(self) -> None:
        q_table, _ = train_q_table(episodes=500, seed=0)
        q_table_before = [row.copy() for row in q_table]

        successes, average_steps, average_reward = evaluate_q_table(
            q_table, episodes=10
        )

        self.assertEqual(successes, 10)
        self.assertEqual(average_steps, 4.0)
        self.assertAlmostEqual(average_reward, 0.97)
        self.assertEqual(q_table, q_table_before)

    def test_invalid_epsilon_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            choose_action_index([0.0, 0.0], 1.1, random.Random(0))


if __name__ == "__main__":
    unittest.main()
