"""
Classic non-contextual UCB for finite-armed bandits.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from algorithms.base_algo import BaseBanditAlgorithm


class UCB(BaseBanditAlgorithm):
    """UCB with empirical means and logarithmic exploration bonus."""

    def __init__(self, n_actions: int, alpha: float = 1.0):
        super().__init__(n_actions=n_actions)
        if n_actions < 2:
            raise ValueError("n_actions must be >= 2")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")

        self.alpha = float(alpha)

        self.t = 0
        self.counts = np.zeros(n_actions, dtype=np.int64)
        self.reward_sums = np.zeros(n_actions, dtype=np.float64)

    def select_action(self, context: np.ndarray) -> int:
        for action in range(self.n_actions):
            if self.counts[action] == 0:
                return action

        effective_t = max(self.t, 2)

        means = self.reward_sums / self.counts
        bonus = self.alpha * np.sqrt(2.0 * np.log(effective_t) / self.counts)
        scores = means + bonus
        return int(np.argmax(scores))

    def update(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.t += 1
        self.counts[action] += 1
        self.reward_sums[action] += float(reward)
