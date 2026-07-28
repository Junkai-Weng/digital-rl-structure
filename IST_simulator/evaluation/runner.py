"""
Evaluator Runner for Bandit Algorithms.
This module defines the `EvaluatorRunner` class, which is responsible for orchestrating 
the interaction between a bandit environment and a bandit algorithm. It manages the 
execution of the experiment, including the action selection, environment stepping, 
algorithm updating, and performance metric recording. The runner ensures that the 
evaluation process is consistent and that key metrics such as cumulative reward and 
regret are computed correctly for analysis.
"""
import numpy as np
from typing import Dict, List
from envs.base_env import BaseBanditEnv
from algorithms.base_algo import BaseBanditAlgorithm

class EvaluatorRunner:
    """
    Bandit Evaluator Runner
    """
    def __init__(self, env: BaseBanditEnv, algo: BaseBanditAlgorithm, n_steps: int):
        self.env = env
        self.algo = algo
        self.n_steps = n_steps
        
        # record history for metrics calculation
        self.rewards_history: List[float] = []
        self.optimal_rewards_history: List[float] = []
        self.regret_history: List[float] = []
        
        # record total interactions (including discarded steps in offline evaluation)
        self.total_interactions = 0 
        
    def run_experiment(self) -> Dict[str, np.ndarray]:
        """
        Run the core interaction loop. Compatible with online environments and offline rejection sampling environments.
        """
        context = self.env.reset()
        valid_step_count = 0  
        done = False
        
        while not done and valid_step_count < self.n_steps:
            self.total_interactions += 1
            
            # 1. Algorithm selects an action
            action = self.algo.select_action(context)
            
            # 2. Environment transitions
            next_context, reward, done, info = self.env.step(action)
            
            # 3. Check if the step is "valid" (compatible with rejection sampling)
            # If the environment did not pass the 'accepted' field (e.g., pure synthetic environments), default to True
            is_accepted = info.get('accepted', True)
            
            if is_accepted:
                # Only accepted interactions are used for algorithm updates and metric recording
                self.algo.update(context, action, reward, info)
                self.rewards_history.append(reward)
                valid_step_count += 1
                
                # Compatible with Regret calculation (only if the environment provides the optimal reward)
                if 'optimal_reward' in info and 'expected_reward' in info:
                    instant_regret = info['optimal_reward'] - info['expected_reward']
                    self.regret_history.append(instant_regret)
                    self.optimal_rewards_history.append(info['optimal_reward'])
            
            # 4. Move to the next context 
            context = next_context
            
        return self._compute_metrics()
        
    def _compute_metrics(self) -> Dict[str, np.ndarray]:
        """Compute final metrics"""
        metrics = {
            "valid_steps": np.array([len(self.rewards_history)]),
            "total_interactions": np.array([self.total_interactions]),
            "cumulative_reward": np.cumsum(self.rewards_history) if self.rewards_history else np.array([])
        }
        
        if self.regret_history:
            metrics["cumulative_regret"] = np.cumsum(self.regret_history)
            
        return metrics