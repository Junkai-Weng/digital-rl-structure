from __future__ import annotations
"""Unified offline dataset container for contextual bandit logs.

The class standardizes how datasets are represented across generators and
environments, and provides split + serialization helpers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class OfflineBanditDataset:
    """Canonical offline dataset schema used by the data module.

    Required fields are `contexts`, `actions`, and `rewards`.
    Optional fields (`optimal_rewards`, `expected_rewards`, `metadata`) are
    preserved when present for richer analysis.
    """

    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    optimal_rewards: Optional[np.ndarray] = None
    expected_rewards: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate array shapes and enforce a consistent in-memory format."""
        self.contexts = np.asarray(self.contexts)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards)

        if self.contexts.ndim != 2:
            raise ValueError(f"contexts must be 2D, got shape={self.contexts.shape}")
        if self.actions.ndim != 1:
            raise ValueError(f"actions must be 1D, got shape={self.actions.shape}")
        if self.rewards.ndim != 1:
            raise ValueError(f"rewards must be 1D, got shape={self.rewards.shape}")

        n = self.contexts.shape[0]
        if len(self.actions) != n or len(self.rewards) != n:
            raise ValueError("contexts, actions, rewards must have same number of rows")

        if self.optimal_rewards is not None:
            self.optimal_rewards = np.asarray(self.optimal_rewards)
            if self.optimal_rewards.shape[0] != n:
                raise ValueError("optimal_rewards length mismatch")

        if self.expected_rewards is not None:
            self.expected_rewards = np.asarray(self.expected_rewards)
            if self.expected_rewards.shape[0] != n:
                raise ValueError("expected_rewards length mismatch")

    @property
    def n_samples(self) -> int:
        """Number of logged interactions."""
        return int(self.contexts.shape[0])

    @property
    def context_dim(self) -> int:
        """Context feature dimension."""
        return int(self.contexts.shape[1])

    @property
    def n_actions(self) -> int:
        """Inferred number of actions (`max(action)+1`)."""
        return int(np.max(self.actions)) + 1 if self.actions.size > 0 else 0

    def subset(self, indices: np.ndarray) -> "OfflineBanditDataset":
        """Return a row subset while preserving optional fields."""
        optimal = self.optimal_rewards[indices] if self.optimal_rewards is not None else None
        expected = self.expected_rewards[indices] if self.expected_rewards is not None else None
        return OfflineBanditDataset(
            contexts=self.contexts[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            optimal_rewards=optimal,
            expected_rewards=expected,
            metadata=dict(self.metadata or {}),
        )

    def train_test_split(self, test_size: float = 0.2, seed: int = 42, shuffle: bool = True) -> Tuple["OfflineBanditDataset", "OfflineBanditDataset"]:
        """Split the dataset into train/test subsets.

        Args:
            test_size: Fraction of rows assigned to the test split.
            seed: Random seed used when `shuffle=True`.
            shuffle: Whether to shuffle before splitting.
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")

        n = self.n_samples
        if n < 2:
            raise ValueError("Need at least 2 samples to split")

        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)

        test_n = max(1, int(round(n * test_size)))
        if test_n >= n:
            test_n = n - 1

        test_idx = idx[:test_n]
        train_idx = idx[test_n:]
        return self.subset(train_idx), self.subset(test_idx)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset into a NumPy-serializable dictionary."""
        payload: Dict[str, Any] = {
            "contexts": self.contexts,
            "actions": self.actions,
            "rewards": self.rewards,
        }
        if self.optimal_rewards is not None:
            payload["optimal_rewards"] = self.optimal_rewards
        if self.expected_rewards is not None:
            payload["expected_rewards"] = self.expected_rewards
        if self.metadata is not None:
            payload["metadata"] = np.array([self.metadata], dtype=object)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OfflineBanditDataset":
        """Build a dataset from a dictionary payload."""
        meta = data.get("metadata")
        if isinstance(meta, np.ndarray) and meta.size == 1:
            meta = meta.item()

        return cls(
            contexts=data["contexts"],
            actions=data["actions"],
            rewards=data["rewards"],
            optimal_rewards=data.get("optimal_rewards"),
            expected_rewards=data.get("expected_rewards"),
            metadata=meta,
        )

    def save_npz(self, path: str) -> None:
        """Persist dataset to `.npz` file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **self.to_dict())

    @classmethod
    def load_npz(cls, path: str) -> "OfflineBanditDataset":
        """Load dataset from `.npz` file."""
        with np.load(path, allow_pickle=True) as data:
            payload = {k: data[k] for k in data.files}
        return cls.from_dict(payload)