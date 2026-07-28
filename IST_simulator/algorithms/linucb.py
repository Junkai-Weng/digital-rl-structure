"""
Linear Upper Confidence Bound (LinUCB) Algorithm.

This module implements the LinUCB algorithm with disjoint linear models for each arm.
"""

import numpy as np
from typing import Dict, Any, Optional
from algorithms.base_algo import BaseBanditAlgorithm

class LinUCB(BaseBanditAlgorithm):
    """
    LinUCB Algorithm (Disjoint Models).
Each action maintains its own ridge regression model to estimate rewards and compute confidence bounds.
    """
    def __init__(self, n_actions: int, context_dim: int, alpha: float = 1.0, l2_reg: float = 1.0):
        """
        Initialize the LinUCB algorithm.
        
        Args:
            n_actions: Number of actions.
            context_dim: Dimension of context features.
            alpha: Exploration parameter. A larger alpha results in a wider confidence interval and a stronger exploration tendency.
            l2_reg: L2 regularization coefficient for ridge regression (corresponding to lambda in the formula).
        """
        super().__init__(n_actions)
        self.context_dim = context_dim
        self.alpha = alpha
        self.l2_reg = l2_reg
        
        # For efficient updates, we maintain the inverse of A_a (the covariance matrix) for each action, and the b_a vector.
        # A_inv shape: (n_actions, context_dim, context_dim)
        # b_a shape: (n_actions, context_dim)
        
        # Initialize A_inv as (1/l2_reg) * I
        self.A_inv = np.array([np.eye(context_dim) / self.l2_reg for _ in range(n_actions)])
        # Initialize b_a as zero vector
        self.b = np.zeros((n_actions, context_dim))

    def select_action(self, context: np.ndarray) -> int:
        """
        Compute the UCB values for all actions and select the action with the maximum value.
        """
        ucb_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # Get the inverse matrix and b vector for the current action
            A_inv_a = self.A_inv[a]
            b_a = self.b[a]
            
            # 1. Compute the parameter estimate theta_hat
            theta_hat = A_inv_a @ b_a
            
            # 2. Compute the expected reward mean
            expected_reward = context.T @ theta_hat
            
            # 3. Compute the confidence bound
            cb = self.alpha * np.sqrt(context.T @ A_inv_a @ context)
            
            # 4. Aggregate to get the UCB score
            ucb_values[a] = expected_reward + cb
            
        # Tie-breaking: If multiple actions have the same UCB value, randomly select one
        max_ucb = np.max(ucb_values)
        best_actions = np.where(ucb_values == max_ucb)[0]
        
        return int(best_actions[0])

    def update(self, context: np.ndarray, action: int, reward: float, info: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the model based on the environment feedback using the Sherman-Morrison formula.
        """
        x = context.reshape(-1, 1) # Convert to column vector (context_dim, 1)
        
        # 1. Update b_a
        self.b[action] += (reward * context)
        
        # 2. Update A_a's inverse matrix using the Sherman-Morrison formula
        # Formula: (A + u*v^T)^-1 = A^-1 - (A^-1 * u * v^T * A^-1) / (1 + v^T * A^-1 * u)
        # Here u = v = x
        A_inv_a = self.A_inv[action]
        numerator = A_inv_a @ x @ x.T @ A_inv_a
        denominator = 1.0 + float(x.T @ A_inv_a @ x)
        
        self.A_inv[action] = A_inv_a - (numerator / denominator)