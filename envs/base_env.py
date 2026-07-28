"""
Defines the abstract base class for Contextual Bandit environments.
This module specifies the standard interface that all bandit environments (whether synthetic or 
dataset-based) must adhere to. It ensures that different environment sources can be utilized 
by a unified evaluation framework.
Core functionalities include:
1. `__init__`: Initializes action space size and context dimensions.
2. `reset`: Resets the environment and returns the initial context.
3. `get_context`: Retrieves the current context vector.
4. `step`: Executes an action and returns the next context, reward, completion flag, and auxiliary info.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseBanditEnv(ABC):
    """
    Bandit Environment abstract base class defining the interface for contextual bandit environments.
    All specific bandit environments (synthetic or data-driven) should inherit from this class and implement the abstract methods.
    """
    def __init__(self, n_actions: int, context_dim: int):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.current_context = None

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment state, typically called at the start of a new experiment (Run).
        
        Returns:
            np.ndarray: Initial context vector.
        """
        pass

    @abstractmethod
    def get_context(self) -> np.ndarray:
        """
        Get the current context. In a standard Contextual Bandit, the context is refreshed after each step.
        
        Returns:
            np.ndarray: Current context vector.
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Receive the action from the algorithm and return feedback.
        
        Args:
            action (int): The action index chosen by the algorithm.
            
        Returns:
            Tuple containing:
                - next_context (np.ndarray): The context for the next step (usually a new context sampled independently in Bandit problems).
                - reward (float): The reward obtained from taking the action.
                - done (bool): Indicates whether the experiment has ended (e.g., reached maximum steps).
                - info (Dict): Additional information dictionary (e.g., ground truth best action/optimal reward for regret calculation).
        """
        pass