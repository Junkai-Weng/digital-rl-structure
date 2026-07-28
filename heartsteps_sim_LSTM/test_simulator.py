# test_simulator.py

from heartstep_simulator import HeartStepsSimulator
import matplotlib.pyplot as plt

def main():
    sim = HeartStepsSimulator(T=90, sigma_noise=30.0, random_state=123)

    # Simulate one user
    traj = sim.simulate_user(policy="random", p_action=0.5)

    S = traj["S"]   # shape (T, 6)
    A = traj["A"]   # shape (T,)
    R = traj["R"]   # shape (T,)

    print("S shape:", S.shape)
    print("A shape:", A.shape)
    print("R shape:", R.shape)
    print("First 5 actions:", A[:5])
    print("First 5 rewards:", R[:5])

    # Plot reward over time to see if it looks reasonable
    plt.plot(R)
    plt.xlabel("Day")
    plt.ylabel("Reward (proxy for activity)")
    plt.title("Simulated HeartSteps-like Rewards")
    plt.show()

if __name__ == "__main__":
    main()