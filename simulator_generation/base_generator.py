"""
Base Simulator Generator Module for Bandit Environments.
This module defines the abstract base class `BaseSimulatorGenerator` which serves as 
the foundation for implementing various simulator generators that can create interactive 
bandit environments from offline data. The generator is responsible for fitting a data 
generation process (DGP) based on the provided offline data and then producing environment 
instances that conform to the `BaseBanditEnv` interface. This allows for a standardized 
way to convert static datasets into dynamic environments suitable for evaluating bandit algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from envs.base_env import BaseBanditEnv

class BaseSimulatorGenerator(ABC):
    """
    Simulator generator abstract base class.
    Responsible for receiving offline data, fitting distributions, and outputting interactive Bandit environments.
    """
    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, offline_data: Any) -> None:
        """
        Fit the data generation process (DGP) based on offline data (or the distribution settings of synthetic data).
        
        Args:
            offline_data: Offline log data. The specific type depends on the subsequent implementation (possibly a pandas DataFrame or Dict).
        """
        pass

    @abstractmethod
    def generate_env(self, env_params: Dict[str, Any] = None) -> BaseBanditEnv:
        """
        Instantiate and return an environment object.
        
        Args:
            env_params (Dict, optional): Parameters to override the default environment configuration.
            
        Returns:
            BaseBanditEnv: An instance conforming to the BaseBanditEnv interface.
        """
        pass
        
    @abstractmethod
    def sample_simulator_distribution(self, n_simulators: int) -> List[BaseBanditEnv]:
        """
        Sample multiple perturbed environments based on data uncertainty for robustness evaluation.
        
        Args:
            n_simulators (int): The number of simulators to generate.
            
        Returns:
            List[BaseBanditEnv]: A list containing multiple environment instances.
        """
        pass