"""Linear Thompson Sampling for contextual bandits.

Model:
    For each action a:
        r = x^T theta_a + noise

Bayesian linear regression posterior with Gaussian prior:
    theta_a ~ N(0, (1/lam) I)

Posterior:
    A_a = lam I + sum_{t: a_t=a} x_t x_t^T
    b_a = sum_{t: a_t=a} r_t x_t
    mu_a = A_a^{-1} b_a
    theta_sample ~ N(mu_a, sigma^2 * A_a^{-1})

Action selection:
    pick argmax_a x^T theta_sample_a
"""

from typing import Dict, Any, Optional

import numpy as np

from algorithms.base_algo import BaseBanditAlgorithm


class LinTS(BaseBanditAlgorithm):
    """Disjoint linear Thompson Sampling with Gaussian posterior per action."""

    def __init__(self, n_actions: int, context_dim: int, lam: float = 1.0, sigma: float = 1.0, seed: int = 42):
        super().__init__(n_actions=n_actions)
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if lam <= 0:
            raise ValueError("lam must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.context_dim = int(context_dim)
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

        self.A = [self.lam * np.eye(self.context_dim, dtype=np.float64) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.context_dim, dtype=np.float64) for _ in range(self.n_actions)]
        self.A_inv = [np.linalg.inv(self.A[a]) for a in range(self.n_actions)]

    def select_action(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(f"context dim mismatch: got {x.shape[0]}, expected {self.context_dim}")

        best_action = 0
        best_value = -np.inf
        for action in range(self.n_actions):
            inv_mat = self.A_inv[action]
            mu = inv_mat @ self.b[action]

            try:
                chol = np.linalg.cholesky(inv_mat)
            except np.linalg.LinAlgError:
                chol = np.linalg.cholesky(inv_mat + 1e-9 * np.eye(self.context_dim, dtype=np.float64))

            theta_sample = mu + self.sigma * (chol @ self.rng.normal(size=self.context_dim))
            sampled_value = float(x @ theta_sample)
            if sampled_value > best_value:
                best_value = sampled_value
                best_action = action

        return int(best_action)

    def update(self, context: np.ndarray, action: int, reward: float, info: Optional[Dict[str, Any]] = None) -> None:
        chosen_action = int(action)
        if not (0 <= chosen_action < self.n_actions):
            raise ValueError(f"action out of range: {chosen_action}")

        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(f"context dim mismatch: got {x.shape[0]}, expected {self.context_dim}")

        self.A[chosen_action] += np.outer(x, x)
        self.b[chosen_action] += float(reward) * x
        self.A_inv[chosen_action] = np.linalg.inv(self.A[chosen_action])