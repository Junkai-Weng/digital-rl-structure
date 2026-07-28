

import numpy as np


class HeartStepsSimulator:
    """
    - Each day t, we choose action A_t in {0,1}.
    - Reward R_t = baseline_gamma_t + A_t * (Z_t^T delta) + noise.
    """

    def __init__(
        self,
        T: int = 90,
        sigma_noise: float = 30.0,
        random_state: int | None = None,
    ):
        """
            T: number of time steps (days) per user
            sigma_noise: standard deviation of Gaussian noise
        """
        self.T = T
        self.sigma_noise = sigma_noise
        self.rng = np.random.RandomState(random_state)

        # Precompute baseline and treatment-effect features
        self._setup_baseline_and_features()

    def _setup_baseline_and_features(self):
        """
        Set up:
        - gamma_t (baseline)
        - Z_t (treatment-effect features)
        - delta (treatment-effect coefficients)
        """

        T = self.T

        # Baseline gamma_t: decreases linearly from 125 to 50 over T days
        self.gamma = np.linspace(125.0, 50.0, T)  # shape (T,)

        # Treatment-effect features Z_t as in run_exp.py for env='mobile'
        # Z_t = [1, (t-1)/45, ((t-1)/45)^2]
        self.Z = np.array([[1.0, (t - 1) / 45.0, ((t - 1) / 45.0) ** 2] for t in range(T)])

        # Treatment-effect coefficients delta (copied from run_exp.py logic)
        x2 = -6.0 / (((89.0 / 45.0) ** 2) * -9.0 / 8.0 + 89.0 / 45.0)
        x1 = -x2 * 9.0 / 8.0
        self.delta = np.array([6.0, x2, x1])  # shape (3,)

    def simulate_user(self, policy="random", p_action: float = 0.5):
        """
        Simulate one user trajectory of length T.

        Args:
            policy: currently only 'random' is supported.
            p_action: probability of A_t = 1 under the random policy.

        Returns:
            A dict with:
                S: array of shape (T, s_dim)  -- state features
                A: array of shape (T,)        -- actions (0 or 1)
                R: array of shape (T,)        -- rewards (float)
        """
        T = self.T

        A = np.zeros(T, dtype=int)
        R = np.zeros(T, dtype=float)

        # We'll define state S_t as:
        #   S_t = [t_norm, Z_t (3 dims), gamma_t, prev_R]
        #       = [1 + 3 + 1 + 1 = 6 dims]
        S = np.zeros((T, 6), dtype=float)

        prev_R = 0.0  # use 0 as R_{0}

        for t in range(T):
            # 1. Choose action A_t
            if policy == "random":
                a_t = self.rng.binomial(1, p_action)
            else:
                raise NotImplementedError("Only 'random' policy is implemented for now.")

            # 2. Compute reward
            gamma_t = self.gamma[t]          # baseline
            z_t = self.Z[t]                  # shape (3,)
            treatment_effect = np.dot(z_t, self.delta)
            noise = self.rng.normal(loc=0.0, scale=self.sigma_noise)

            r_t = gamma_t + a_t * treatment_effect + noise

            # 3. Build state S_t
            t_norm = (t + 1) / T             # normalize time to (0,1]
            s_t = np.array([
                t_norm,
                z_t[0],
                z_t[1],
                z_t[2],
                gamma_t,
                prev_R,
            ], dtype=float)

            # 4. Save
            S[t] = s_t
            A[t] = a_t
            R[t] = r_t

            # 5. Update previous reward
            prev_R = r_t

        return {"S": S, "A": A, "R": R}

    def simulate_dataset(self, num_users: int = 20, policy="random", p_action: float = 0.5):
        """
        Simulate a dataset with multiple users.

        Returns:
            A list of length num_users, each element is a dict(S, A, R)
        """
        dataset = []
        for n in range(num_users):
            traj = self.simulate_user(policy=policy, p_action=p_action)
            dataset.append(traj)
        return dataset