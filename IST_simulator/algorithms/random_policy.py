
import numpy as np

class RandomPolicy:
    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, x):
        return int(self.rng.integers(self.n_actions))

    def update(self, x, a, r):
        pass