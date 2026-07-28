"""
Rejection Sampling Environment - based on Li et al. (2010) logic for offline evaluation. 
It reads offline data sequentially like playing a tape recorder.
"""
import numpy as np
import pickle
from typing import Tuple, Dict, Any
from pathlib import Path
from envs.base_env import BaseBanditEnv

class RejectionSamplingEnv(BaseBanditEnv):
    """
    Rejection Sampling Environment for Offline Bandit Evaluation
    """
    def __init__(self, dataset: Any):
        """
        Args:
            dataset: Can be either:
                - a dict with keys 'contexts', 'actions', 'rewards'
                - an object with attributes `contexts`, `actions`, `rewards`

        This makes the environment compatible with both dictionary-based
        logs and dataset class instances.
        """
        # Support both dict-style and object-style datasets.
        if isinstance(dataset, dict):
            self.contexts = dataset['contexts']
            self.actions = dataset['actions']
            self.rewards = dataset['rewards']
            self.optimal_rewards = dataset.get('optimal_rewards')
            self.expected_rewards = dataset.get('expected_rewards')
        else:
            self.contexts = dataset.contexts
            self.actions = dataset.actions
            self.rewards = dataset.rewards
            self.optimal_rewards = getattr(dataset, 'optimal_rewards', None)
            self.expected_rewards = getattr(dataset, 'expected_rewards', None)

        self.n_samples = len(self.actions)
        self.current_idx = 0
        
        # Automatically infer dimensions and initialize parent class
        context_dim = self.contexts.shape[1]
        n_actions = len(np.unique(self.actions))
        super().__init__(n_actions=n_actions, context_dim=context_dim)

    def reset(self) -> np.ndarray:
        self.current_idx = 0
        self.current_context = self.contexts[self.current_idx]
        return self.current_context

    def get_context(self) -> np.ndarray:
        return self.current_context

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.current_idx >= self.n_samples:
            # data exhausted, episode ends - return dummy values
            return self.current_context, 0.0, True, {"accepted": False}
            
        logged_action = self.actions[self.current_idx]
        logged_reward = float(self.rewards[self.current_idx])
        
        # Core logic: Compare actions
        accepted = (action == logged_action)
        info = {"accepted": accepted}
        if self.optimal_rewards is not None:
            info["optimal_reward"] = float(self.optimal_rewards[self.current_idx])
            info["expected_reward"] = float(self.expected_rewards[self.current_idx])
        # Regardless of acceptance, the data pointer moves down one row
        self.current_idx += 1
        done = (self.current_idx >= self.n_samples)
        
        self.current_context = self.contexts[self.current_idx] if not done else None
        reward_to_return = logged_reward if accepted else 0.0
        
        return self.current_context, reward_to_return, done, info

    def save_simulator(self, output_path: str) -> None:
        # Persist a fully configured environment for reproducible offline runs.
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)


def build_simulator(dataset: Any, output_path: str | None = None) -> RejectionSamplingEnv:
    """Create a rejection-sampling environment and optionally serialize it."""
    env = RejectionSamplingEnv(dataset)
    if output_path is not None:
        env.save_simulator(output_path)
    return env


def load_simulator(path: str) -> RejectionSamplingEnv:
    """Load a serialized simulator and validate its runtime type."""
    with open(path, "rb") as handle:
        simulator = pickle.load(handle)
    if not isinstance(simulator, RejectionSamplingEnv):
        raise TypeError("Loaded object is not a RejectionSamplingEnv")
    return simulator