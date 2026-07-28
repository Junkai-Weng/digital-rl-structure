
"""
Linear Thompson Sampling for contextual bandits

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

  a = algo.select_action(x)
  algo.update(x, a, r)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LinearTS:
    n_actions: int
    context_dim: int
    lam: float = 1.0          # ridge / prior precision
    sigma: float = 1.0        # sampling scale (exploration)
    seed: int = 0

    def __post_init__(self):
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.lam <= 0:
            raise ValueError("lam must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

        self.rng = np.random.default_rng(self.seed)

        # Per-action posterior state: A (dxd), b (d,)
        self.A = [self.lam * np.eye(self.context_dim, dtype=np.float64) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.context_dim, dtype=np.float64) for _ in range(self.n_actions)]

        self.A_inv = [np.linalg.inv(self.A[a]) for a in range(self.n_actions)]

    def select_action(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(f"x has dim {x.shape[0]}, expected {self.context_dim}")

        best_a = 0
        best_val = -np.inf

        # Sample theta for each arm and compute sampled value
        for a in range(self.n_actions):
            Ainv = self.A_inv[a]
            mu = Ainv @ self.b[a]

            # Sample theta ~ N(mu, sigma^2 * Ainv)
            # Use Cholesky for stability
            try:
                L = np.linalg.cholesky(Ainv)
            except np.linalg.LinAlgError:
                Ainv_j = Ainv + 1e-9 * np.eye(self.context_dim)
                L = np.linalg.cholesky(Ainv_j)

            z = self.rng.normal(size=self.context_dim)
            theta_sample = mu + self.sigma * (L @ z)

            val = float(x @ theta_sample)
            if val > best_val:
                best_val = val
                best_a = a

        return int(best_a)

    def update(self, x: np.ndarray, action: int, reward: float):
        a = int(action)
        if not (0 <= a < self.n_actions):
            raise ValueError(f"action {a} out of range [0, {self.n_actions-1}]")

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(f"x has dim {x.shape[0]}, expected {self.context_dim}")

        r = float(reward)

        # Bayesian linear regression update:
        # A_a <- A_a + x x^T
        # b_a <- b_a + r x
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x

        # Update cached inverse (simple but OK for small d)
        self.A_inv[a] = np.linalg.inv(self.A[a])