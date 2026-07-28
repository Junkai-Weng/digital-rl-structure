
"""
SoftmaxLinear: 

- Compute predicted reward per arm: mu_a = x^T theta_a
- Choose action via softmax over mu / tau:
    p(a) = exp(mu_a / tau) / sum_b exp(mu_b / tau)

"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from algorithms.base_algo import BaseBanditAlgorithm


class SoftmaxLinear(BaseBanditAlgorithm):
    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        tau: float = 0.1,
        l2_reg: float = 1.0,
        seed: int = 0,
    ):
        """
        Args:
            n_actions: number of arms K
            context_dim: feature dimension d
            tau: temperature (>0). Smaller -> greedier; larger -> more exploratory
            l2_reg: ridge regularization lambda (>0)
            seed: RNG seed for action sampling
        """
        super().__init__(n_actions)

        if float(tau) <= 0.0:
            raise ValueError("tau must be > 0.")
        if float(l2_reg) <= 0.0:
            raise ValueError("l2_reg must be > 0.")

        self.context_dim = int(context_dim)
        self.tau = float(tau)
        self.l2_reg = float(l2_reg)

        # Per-arm inverse covariance and b vector
        self.A_inv = np.array([np.eye(self.context_dim) / self.l2_reg for _ in range(n_actions)])
        self.b = np.zeros((n_actions, self.context_dim), dtype=float)

        self.rng = np.random.default_rng(int(seed))

    def _predict_means(self, x: np.ndarray) -> np.ndarray:
        """Return mu_a = x^T theta_hat_a for all arms."""
        mu = np.zeros(self.n_actions, dtype=float)
        for a in range(self.n_actions):
            theta_hat = self.A_inv[a] @ self.b[a]
            mu[a] = float(x @ theta_hat)
        return mu

    def select_action(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(f"context dim mismatch: expected {self.context_dim}, got {x.shape[0]}")

        mu = self._predict_means(x)

        # Softmax over logits = mu / tau
        logits = mu / self.tau

        # Numerically stable softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Sample action
        a = int(self.rng.choice(self.n_actions, p=probs))
        return a

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

        x = np.asarray(context, dtype=float).reshape(-1, 1) 
        if x.shape[0] != self.context_dim:
            raise ValueError(f"context dim mismatch: expected {self.context_dim}, got {x.shape[0]}")

        r = float(reward)

        # Update b
        self.b[a] += (r * x.reshape(-1))

        # Sherman–Morrison update of A_inv
        A_inv_a = self.A_inv[a]
        denom = 1.0 + float(x.T @ A_inv_a @ x)
        numer = A_inv_a @ x @ x.T @ A_inv_a
        self.A_inv[a] = A_inv_a - (numer / denom)