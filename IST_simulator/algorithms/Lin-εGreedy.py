
"""
Lin-εGreedy: 

- Disjoint (per-action) ridge regression.
- With prob ε: explore uniformly at random.
- With prob 1-ε: exploit by choosing arm with highest predicted reward.

"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from algorithms.base_algo import BaseBanditAlgorithm


class LinEpsilonGreedy(BaseBanditAlgorithm):
    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        epsilon: float = 0.1,
        l2_reg: float = 1.0,
        seed: int = 0,
    ):
        """
        Args:
            n_actions: number of arms K
            context_dim: feature dimension d
            epsilon: exploration probability in [0,1]
            l2_reg: ridge regularization lambda (>0)
            seed: RNG seed for exploration
        """
        super().__init__(n_actions)

        if not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")
        if float(l2_reg) <= 0.0:
            raise ValueError("l2_reg must be > 0.")

        self.context_dim = int(context_dim)
        self.epsilon = float(epsilon)
        self.l2_reg = float(l2_reg)

        # Per-arm inverse covariance and b vector 
        self.A_inv = np.array([np.eye(self.context_dim) / self.l2_reg for _ in range(n_actions)])
        self.b = np.zeros((n_actions, self.context_dim), dtype=float)

        self.rng = np.random.default_rng(int(seed))

    def select_action(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(f"context dim mismatch: expected {self.context_dim}, got {x.shape[0]}")

        # Explore
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        # Exploit: choose argmax predicted reward
        preds = np.zeros(self.n_actions, dtype=float)
        for a in range(self.n_actions):
            theta_hat = self.A_inv[a] @ self.b[a]
            preds[a] = float(x @ theta_hat)

        return int(np.argmax(preds))

    def update(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        a = int(action)
        if not (0 <= a < self.n_actions):
            raise ValueError(f"action out of range: {a}")

        x = np.asarray(context, dtype=float).reshape(-1, 1)  # (d,1)
        if x.shape[0] != self.context_dim:
            raise ValueError(f"context dim mismatch: expected {self.context_dim}, got {x.shape[0]}")

        r = float(reward)

        # Update b
        self.b[a] += (r * x.reshape(-1))

        # Sherman–Morrison update of A_inv: A_inv <- A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        A_inv_a = self.A_inv[a]
        denom = 1.0 + float(x.T @ A_inv_a @ x)
        numer = A_inv_a @ x @ x.T @ A_inv_a
        self.A_inv[a] = A_inv_a - (numer / denom)