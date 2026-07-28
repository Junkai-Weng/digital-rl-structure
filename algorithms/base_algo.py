"""
Base Algorithm Module for Bandit Algorithms.
This module defines the abstract base class `BaseBanditAlgorithm` which serves as the 
foundation for implementing various bandit algorithms (e.g., LinUCB, Thompson Sampling) 
within the reinforcement learning benchmark structure. It enforces a consistent interface 
for action selection and parameter updating across different algorithm implementations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseBanditAlgorithm(ABC):
    """
    Bandit Algorithm abstract base class defining the interface for contextual bandit algorithms.
    All algorithms (e.g., LinUCB, Thompson Sampling) must inherit from this class.
    """
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    @abstractmethod
    def select_action(self, context: np.ndarray) -> int:
        """
        Select an action based on the current context.
        
        Args:
            context (np.ndarray): Current feature vector.
            
        Returns:
            int: Selected action index.
        """
        pass

    @abstractmethod
    def update(self, context: np.ndarray, action: int, reward: float, info: Optional[Dict[str, Any]] = None) -> None:
        """
        Receive environment feedback and update the algorithm's internal parameters (e.g., value estimates, confidence intervals).
        
        Args:
            context (np.ndarray): Feature vector at the time of action.
            action (int): The action taken.
            reward (float): The reward received.
            info (Dict, optional): Additional information that may be used for updates.
        """
        pass