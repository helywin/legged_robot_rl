import random
import unittest

from examples.replay_buffer_demo import ReplayBuffer, make_transition


class ReplayBufferDemoTest(unittest.TestCase):
    def test_capacity_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=0)

    def test_full_buffer_evicts_oldest_transition(self) -> None:
        replay_buffer = ReplayBuffer(capacity=2)
        replay_buffer.add(make_transition(0))
        replay_buffer.add(make_transition(1))
        replay_buffer.add(make_transition(2))

        observations = [
            transition.observation for transition in replay_buffer.snapshot()
        ]
        self.assertEqual(observations, [1, 2])

    def test_sample_uses_only_stored_transitions(self) -> None:
        replay_buffer = ReplayBuffer(capacity=4)
        for observation in range(4):
            replay_buffer.add(make_transition(observation))

        sampled = replay_buffer.sample(3, random.Random(7))

        self.assertEqual(len(sampled), 3)
        self.assertEqual(len(set(sampled)), 3)
        self.assertTrue(set(sampled).issubset(set(replay_buffer.snapshot())))

    def test_sample_does_not_remove_transitions(self) -> None:
        replay_buffer = ReplayBuffer(capacity=4)
        for observation in range(4):
            replay_buffer.add(make_transition(observation))
        before = replay_buffer.snapshot()

        replay_buffer.sample(2, random.Random(0))

        self.assertEqual(replay_buffer.snapshot(), before)

    def test_sample_cannot_exceed_stored_count(self) -> None:
        replay_buffer = ReplayBuffer(capacity=4)
        replay_buffer.add(make_transition(0))

        with self.assertRaises(ValueError):
            replay_buffer.sample(2, random.Random(0))


if __name__ == "__main__":
    unittest.main()
