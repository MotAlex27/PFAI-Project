import unittest

import numpy as np
import torch

from agents.dqn import DQNAgent


class DQNAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DQNAgent(
            observation_size=3,
            action_count=2,
            device="cpu",
            seed=13,
            batch_size=2,
            target_update_steps=1,
        )

    def test_action_respects_mask(self) -> None:
        state = np.zeros(3, dtype=np.float32)
        mask = np.array([False, True])
        self.assertEqual(self.agent.act(state, mask, epsilon=1.0), 1)
        self.assertEqual(self.agent.act(state, mask, epsilon=0.0), 1)

    def test_optimization_updates_target_and_returns_finite_loss(self) -> None:
        state = np.zeros(3, dtype=np.float32)
        next_state = np.ones(3, dtype=np.float32)
        mask = np.array([True, False])
        for _ in range(2):
            self.agent.remember(state, 0, 1.0, next_state, mask, False)
        loss = self.agent.optimize()
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(self.agent.optimization_steps, 1)


if __name__ == "__main__":
    unittest.main()
