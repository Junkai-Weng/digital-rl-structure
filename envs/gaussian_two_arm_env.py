"""
Two-armed Gaussian bandit environment with constant gap.

Reward model:
    R_t(a) ~ N(mu_a, sigma^2), a in {0, 1}
with default means mu_0 = Delta, mu_1 = 0 (arm 0 optimal).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.base_env import BaseBanditEnv


class GaussianTwoArmEnv(BaseBanditEnv):
    """Strict 2-arm Gaussian bandit environment (non-contextual)."""

    def __init__(
        self,
        delta: float = 0.2,
        sigma: float = 0.1,
        means: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        super().__init__(n_actions=2, context_dim=1)
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        if means is None:
            if not (0.0 < delta < 1.0):
                raise ValueError("delta must be in (0, 1) when means is not provided")
            means = np.array([delta, 0.0], dtype=float)
        else:
            means = np.asarray(means, dtype=float)
            if means.shape != (2,):
                raise ValueError("means must have shape (2,)")

        self.delta = float(abs(means[0] - means[1]))
        self.sigma = float(sigma)
        self.means = means
        self.rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        self.current_context = np.array([1.0], dtype=np.float32)
        return self.current_context

    def get_context(self) -> np.ndarray:
        if self.current_context is None:
            return np.array([1.0], dtype=np.float32)
        return self.current_context

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")

        reward = float(self.rng.normal(self.means[action], self.sigma))
        optimal_action = int(np.argmax(self.means))
        optimal_reward = float(self.means[optimal_action])

        info = {
            "expected_reward": float(self.means[action]),
            "optimal_action": optimal_action,
            "optimal_reward": optimal_reward,
        }

        self.current_context = np.array([1.0], dtype=np.float32)
        done = False
        return self.current_context, reward, done, info
