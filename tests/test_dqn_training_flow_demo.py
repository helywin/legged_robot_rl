import unittest

from examples.dqn_training_flow_demo import (
    LEFT,
    RIGHT,
    calculate_target_q,
    predict_q_values,
    train_one_transition,
)
from examples.replay_buffer_demo import Transition


class DqnTrainingFlowDemoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.online_weights = {LEFT: 0.2, RIGHT: 0.4}
        self.target_weights = {LEFT: 0.3, RIGHT: 0.5}

    def test_prediction_returns_one_value_per_action(self) -> None:
        q_values = predict_q_values(2.0, self.online_weights)

        self.assertEqual(q_values, {LEFT: 0.4, RIGHT: 0.8})

    def test_non_terminal_target_uses_best_target_network_value(self) -> None:
        transition = Transition(1, RIGHT, 0.1, 2, False, False)

        target_q, next_q_values = calculate_target_q(
            transition, self.target_weights, 0.9
        )

        self.assertEqual(next_q_values, {LEFT: 0.6, RIGHT: 1.0})
        self.assertAlmostEqual(target_q, 1.0)

    def test_terminal_target_uses_reward_only(self) -> None:
        transition = Transition(1, RIGHT, 0.7, 99, True, False)

        target_q, _ = calculate_target_q(
            transition, self.target_weights, 0.9
        )

        self.assertAlmostEqual(target_q, 0.7)

    def test_truncated_transition_can_keep_future_value(self) -> None:
        transition = Transition(1, RIGHT, 0.1, 2, False, True)

        target_q, _ = calculate_target_q(
            transition, self.target_weights, 0.9
        )

        self.assertAlmostEqual(target_q, 1.0)

    def test_training_updates_only_selected_online_action(self) -> None:
        transition = Transition(1, RIGHT, 0.1, 2, False, False)

        updated_online, _ = train_one_transition(
            transition,
            self.online_weights,
            self.target_weights,
            learning_rate=0.2,
            discount_factor=0.9,
        )

        self.assertEqual(updated_online[LEFT], self.online_weights[LEFT])
        self.assertNotEqual(updated_online[RIGHT], self.online_weights[RIGHT])

    def test_training_does_not_modify_target_weights(self) -> None:
        transition = Transition(1, RIGHT, 0.1, 2, False, False)
        target_before = dict(self.target_weights)

        train_one_transition(
            transition,
            self.online_weights,
            self.target_weights,
            learning_rate=0.2,
            discount_factor=0.9,
        )

        self.assertEqual(self.target_weights, target_before)


if __name__ == "__main__":
    unittest.main()
