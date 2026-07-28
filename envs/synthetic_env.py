"""
Linear Contextual Bandit Environment
We start with a simple synthetic environment where the reward is generated from a 
linear model with Gaussian noise.
This environment is designed to be a testbed for evaluating the performance of contextual 
bandit algorithms under controlled conditions. The true parameter vectors (thetas) can 
be set to specific values to test the robustness of algorithms or to create a "simulator 
distribution" by adding small perturbations. The context is sampled independently at each 
step, and the reward is computed based on the action taken and the current context, with 
added noise to simulate real-world uncertainty.
"""


import numpy as np
from typing import Tuple, Dict, Any, Optional
from envs.base_env import BaseBanditEnv

class LinearSyntheticEnv(BaseBanditEnv):
    """
    Linear Contextual Bandit Environment.
    Data Generation Process (DGP): Reward = Context * Theta_a + Noise
    """
    def __init__(self, 
                 n_actions: int = 5, 
                 context_dim: int = 10, 
                 noise_std: float = 0.1,
                 true_thetas: Optional[np.ndarray] = None,
                 seed: int = 42):
        """
        Initialize the synthetic environment.
        
        Args:
            n_actions: Action space size
            context_dim: Dimension of context features
            noise_std: Standard deviation of Gaussian noise for rewards
            true_thetas: True parameter vector matrix with shape (n_actions, context_dim).
                         If not provided, the environment will generate it randomly.
                         This parameter is crucial: to test the robustness of the algorithm or
                         to construct the "simulator distribution", small perturbations can be
                         added to the thetas externally before passing them in.
            seed: Random seed for reproducibility
        """
        super().__init__(n_actions, context_dim)
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        
        # if true_thetas is provided, use it directly; otherwise, generate random thetas
        if true_thetas is not None:
            assert true_thetas.shape == (n_actions, context_dim), "true_thetas dimension error"
            self.true_thetas = true_thetas
        else:
            # Initialize true_thetas from a uniform distribution and perform L2 normalization
            raw_thetas = self.rng.uniform(-1, 1, size=(n_actions, context_dim))
            self.true_thetas = raw_thetas / np.linalg.norm(raw_thetas, axis=1, keepdims=True)
            
        self.current_context = None

    def reset(self) -> np.ndarray:
        """
        Reset the environment. In Bandit, this mainly generates the first Context.
        """
        # We assume Context follows a multivariate standard normal distribution
        self.current_context = self.rng.standard_normal(self.context_dim)
        return self.current_context

    def get_context(self) -> np.ndarray:
        return self.current_context

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action and return the next context, reward, done flag, and info.
        """
        if self.current_context is None:
            raise ValueError("Environment not initialized. Please call reset() first.")
            
        # 1. Compute the expected rewards for all actions (without noise)
        # Shape: (n_actions,)
        expected_rewards = self.true_thetas @ self.current_context
        
        # 2. Compute the actual reward for the taken action (with Gaussian noise)
        reward = expected_rewards[action] + self.rng.normal(0, self.noise_std)
        
        # 3. Extract the Ground Truth for computing Regret
        optimal_action = int(np.argmax(expected_rewards))
        optimal_reward = float(expected_rewards[optimal_action])
        
        info = {
            "expected_reward": float(expected_rewards[action]),
            "optimal_action": optimal_action,
            "optimal_reward": optimal_reward
        }
        
        # 4. Generate the next time step's Context (i.i.d. sampling)
        self.current_context = self.rng.standard_normal(self.context_dim)
        
        # Bandit environments typically do not have the concept of episode termination
        # unless a maximum step count is reached, so done is always False
        done = False
        
        return self.current_context, float(reward), done, info