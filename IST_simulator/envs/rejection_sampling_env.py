
"""

  next_x, reward, done, info = env.step(action)
  info['accepted'] tells the runner whether to update the algorithm / count the step.

- We have offline log tuples (x_i, a_i, r_i).
- The algorithm proposes an action a.
- If a == a_i, we ACCEPT and return reward r_i.
- Otherwise we REJECT (accepted=False) and do not provide a usable reward.
- We always advance the log index so we don't get stuck.

Expected offline_log format (from datasets/ist_loader.py):
  offline_log = {
    "contexts": np.ndarray (N, d) float32,
    "actions":  np.ndarray (N,) int64,
    "rewards":  np.ndarray (N,) float32,
    "sites":    np.ndarray (N,) int64,              ### #might change later!!
    ... (optional) ...
  }

Questions to ask:
- cceptance rates can be low (e.g., ~1/6 for 6 arms)?? maybe try 4 arms
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class RejectionSamplingStats:
    total_interactions: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_interactions == 0:
            return 0.0
        return self.accepted / self.total_interactions


class RejectionSamplingEnv:
    """
    Offline environment based on rejection sampling.

    Behavior:
    - reset() returns the first context (or a random one if shuffle=True).
    - step(action) compares `action` to the logged action at current index.
      * If match: accepted=True, returns logged reward.
      * If mismatch: accepted=False, returns reward=0.0 (ignored by runner).
    - Always increments the internal index.
    - done=True when we've exhausted the log (unless auto_reset=True).

    Parameters
    ----------
    offline_log : dict
        Must contain 'contexts', 'actions', 'rewards'. Optional: 'reward_components'.
    shuffle : bool
        If True, we shuffle the order of rows on reset (good for making the stream i.i.d.).
    seed : int
        RNG seed used if shuffle=True.
    auto_reset : bool
        If True, when the log ends, env resets and continues (done=False).
        If False, env returns done=True at end.
    return_reward_components : bool
        If True and offline_log contains 'reward_components', returns it in info.
    """

    def __init__(
        self,
        offline_log: Dict[str, Any],
        *,
        shuffle: bool = True,
        seed: int = 0,
        auto_reset: bool = False,
        return_reward_components: bool = True,
    ):
        self.offline_log = offline_log
        self.shuffle = shuffle
        self.seed = int(seed)
        self.auto_reset = auto_reset
        self.return_reward_components = return_reward_components

        # Validate required keys
        for k in ("contexts", "actions", "rewards"):
            if k not in offline_log:
                raise KeyError(f"offline_log missing required key '{k}'")

        self.X = np.asarray(offline_log["contexts"])
        self.A = np.asarray(offline_log["actions"], dtype=np.int64)
        self.R = np.asarray(offline_log["rewards"], dtype=np.float32)

        if self.X.ndim != 2:
            raise ValueError(f"offline_log['contexts'] must be 2D array (N,d); got shape {self.X.shape}")
        if self.A.ndim != 1 or self.R.ndim != 1:
            raise ValueError("offline_log['actions'] and ['rewards'] must be 1D arrays")
        if not (len(self.X) == len(self.A) == len(self.R)):
            raise ValueError("contexts/actions/rewards must have same length")

        self.N = len(self.A)
        if self.N == 0:
            raise ValueError("offline_log is empty (N=0)")

        self.reward_components = None
        self.reward_component_names = None
        if return_reward_components and "reward_components" in offline_log:
            self.reward_components = np.asarray(offline_log["reward_components"], dtype=np.float32)
            if self.reward_components.shape[0] != self.N:
                raise ValueError("reward_components must have same number of rows as contexts/actions/rewards")
            self.reward_component_names = list(offline_log.get("reward_component_names", []))

        # Order / index management
        self._rng = np.random.default_rng(self.seed)
        self._order = np.arange(self.N, dtype=np.int64)
        self._idx = 0
        self._current_x = None

        self.stats = RejectionSamplingStats()

    def reset(self) -> np.ndarray:
        """Reset stream and return first context."""
        self.stats = RejectionSamplingStats()
        self._idx = 0

        if self.shuffle:
            self._order = self._rng.permutation(self.N)
        else:
            self._order = np.arange(self.N, dtype=np.int64)

        first_row = self._order[self._idx]
        self._current_x = self.X[first_row]
        return self._current_x

    def _advance(self) -> Tuple[np.ndarray, bool]:
        """
        Advance to next row; return (next_context, done).
        """
        self._idx += 1
        if self._idx >= self.N:
            if self.auto_reset:
                # start over
                self.reset()
                return self._current_x, False
            # end of log
            return self._current_x, True

        row = self._order[self._idx]
        self._current_x = self.X[row]
        return self._current_x, False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step with proposed action.

        Returns:
          next_context, reward, done, info
        """
        if self._current_x is None:
            # User forgot to call reset()
            _ = self.reset()

        # Current row
        row = self._order[self._idx]
        logged_action = int(self.A[row])

        self.stats.total_interactions += 1

        accepted = (int(action) == logged_action)

        if accepted:
            reward = float(self.R[row])
            self.stats.accepted += 1
        else:
            # Reward is not valid when rejected; runner should ignore it (accepted=False).
            reward = 0.0

        # Build info
        info: Dict[str, Any] = {
            "accepted": accepted,
            "logged_action": logged_action,
            "row_index": int(row),
            "acceptance_rate": self.stats.acceptance_rate,
        }

        # expose reward for analysis
        if self.reward_components is not None:
            info["reward_components"] = self.reward_components[row]
            if self.reward_component_names:
                info["reward_component_names"] = self.reward_component_names

        # Advance
        next_context, done = self._advance()

        return next_context, reward, done, info

    # Convenience helpers
    def n_actions(self) -> int:
        """Infer number of actions as max(action)+1."""
        return int(self.A.max() + 1)

    def context_dim(self) -> int:
        return int(self.X.shape[1])


# Tiny test 
if __name__ == "__main__":
    # Minimal fake log
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5)).astype(np.float32)
    A = rng.integers(0, 3, size=100).astype(np.int64)
    R = rng.normal(size=100).astype(np.float32)

    env = RejectionSamplingEnv({"contexts": X, "actions": A, "rewards": R}, shuffle=False)
    x = env.reset()
    total, acc = 0, 0
    for _ in range(50):
        a = int(rng.integers(0, 3))
        x, r, done, info = env.step(a)
        total += 1
        acc += int(info["accepted"])
        if done:
            break
    print("total:", total, "accepted:", acc, "rate:", acc / total if total else 0.0)