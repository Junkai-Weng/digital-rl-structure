"""
Plugin/fitted simulator generator for a 2-arm Gaussian bandit.

Fits mu_hat_a and pooled sigma_hat^2 from offline logs, then generates
interactive GaussianTwoArmEnv instances.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from envs.base_env import BaseBanditEnv
from envs.gaussian_two_arm_env import GaussianTwoArmEnv
from simulator_generation.base_generator import BaseSimulatorGenerator


class GaussianPluginGenerator(BaseSimulatorGenerator):
    """Fit-and-generate simulator for strict 2-arm Gaussian setting."""

    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.mu_hat = None
        self.sigma2_hat = None
        self.arm_counts = None

    def _extract_arrays(self, offline_data: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(offline_data, dict):
            actions = np.asarray(offline_data["actions"])
            rewards = np.asarray(offline_data["rewards"])
            return actions, rewards

        actions = np.asarray(offline_data.actions)
        rewards = np.asarray(offline_data.rewards)
        return actions, rewards

    def fit(self, offline_data: Any) -> None:
        actions, rewards = self._extract_arrays(offline_data)

        if actions.ndim != 1 or rewards.ndim != 1 or len(actions) != len(rewards):
            raise ValueError("actions/rewards must be 1D arrays with same length")

        unique_actions = np.unique(actions)
        if not np.array_equal(np.sort(unique_actions), np.array([0, 1])):
            raise ValueError("GaussianPluginGenerator expects exactly two actions {0,1}")

        arm_counts = np.array([(actions == a).sum() for a in (0, 1)], dtype=np.int64)
        if np.any(arm_counts == 0):
            raise ValueError("Both actions must appear at least once in offline data")

        mu_hat = np.array([rewards[actions == a].mean() for a in (0, 1)], dtype=float)

        residuals = np.empty_like(rewards, dtype=float)
        for a in (0, 1):
            mask = actions == a
            residuals[mask] = rewards[mask] - mu_hat[a]

        dof = max(len(rewards) - 2, 1)
        sigma2_hat = float(np.sum(residuals ** 2) / dof)
        sigma2_hat = max(sigma2_hat, 1e-10)

        self.mu_hat = mu_hat
        self.sigma2_hat = sigma2_hat
        self.arm_counts = arm_counts
        self.is_fitted = True

    def generate_env(self, env_params: Dict[str, Any] | None = None) -> BaseBanditEnv:
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before generate_env(...)")

        env_params = env_params or {}
        means = np.asarray(env_params.get("means", self.mu_hat), dtype=float)
        sigma = float(env_params.get("sigma", np.sqrt(self.sigma2_hat)))
        seed = int(env_params.get("seed", self.rng.integers(1, 10_000_000)))

        return GaussianTwoArmEnv(means=means, sigma=sigma, seed=seed)

    def sample_simulator_distribution(self, n_simulators: int) -> List[BaseBanditEnv]:
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before sample_simulator_distribution(...)")
        if n_simulators <= 0:
            raise ValueError("n_simulators must be > 0")

        sigma_hat = np.sqrt(self.sigma2_hat)
        stderr = sigma_hat / np.sqrt(np.maximum(self.arm_counts, 1))

        simulators: List[BaseBanditEnv] = []
        for _ in range(n_simulators):
            sampled_means = self.rng.normal(loc=self.mu_hat, scale=stderr)
            sampled_sigma = max(float(self.rng.normal(loc=sigma_hat, scale=0.05 * sigma_hat)), 1e-6)
            simulators.append(
                GaussianTwoArmEnv(
                    means=sampled_means,
                    sigma=sampled_sigma,
                    seed=int(self.rng.integers(1, 10_000_000)),
                )
            )
        return simulators
