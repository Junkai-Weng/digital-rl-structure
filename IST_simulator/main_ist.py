# main_ist.py
import numpy as np

from datasets.ist_loader import default_config, load_ist_offline_logs
from envs.rejection_sampling_env import RejectionSamplingEnv

from algorithms.random_policy import RandomPolicy
from algorithms.linucb import LinUCB
from algorithms.linear_ts import LinearTS


def filter_log_to_site(log: dict, site_id: int) -> dict:
    mask = (log["sites"] == site_id)
    out = {
        "contexts": log["contexts"][mask],
        "actions": log["actions"][mask],
        "rewards": log["rewards"][mask],
        "sites": log["sites"][mask],
    }

    if "reward_components" in log and log["reward_components"] is not None:
        out["reward_components"] = log["reward_components"][mask]
        out["reward_component_names"] = log.get("reward_component_names", None)
    return out


def run_on_site(site_log: dict, algo, n_accepted_steps: int = 200, seed: int = 0):
    env = RejectionSamplingEnv(site_log, shuffle=False, seed=seed, auto_reset=False)
    x = env.reset()

    rewards = []
    accepted = 0
    total = 0
    done = False

    while (accepted < n_accepted_steps) and (not done):
        x_prev = x
        a = algo.select_action(x_prev)
        x, r, done, info = env.step(a)
        total += 1

        if info["accepted"]:
            algo.update(x_prev, a, r)  
            rewards.append(r)
            accepted += 1

    mean_reward = float(np.mean(rewards)) if rewards else float("nan")
    acceptance_rate = accepted / total if total > 0 else 0.0
    return mean_reward, acceptance_rate, accepted, total


def summarize(values: list[float], name: str):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    print(f"\n{name}:")
    print(f"  #sites: {len(arr)}")
    print(f"  mean:   {arr.mean():.4f}")
    print(f"  median: {np.median(arr):.4f}")
    print(f"  p10:    {np.quantile(arr, 0.10):.4f}")
    print(f"  p90:    {np.quantile(arr, 0.90):.4f}")


def run_phase(phase_name: str, phase_log: dict, phase_sites, make_algo_fn, n_steps: int, seed: int):
    rewards = []
    acc_rates = []

    for site_id in phase_sites:
        site_log = filter_log_to_site(phase_log, int(site_id))
        if len(site_log["actions"]) == 0:
            continue

        algo = make_algo_fn(int(site_id))
        mean_r, acc_rate, acc, total = run_on_site(site_log, algo, n_accepted_steps=n_steps, seed=seed)

        rewards.append(mean_r)
        acc_rates.append(acc_rate)

    return rewards, acc_rates


def main():
    cfg = default_config()
    train_log, test_log, train_sites, test_sites = load_ist_offline_logs("data/IST_corrected.txt", cfg)

    K = len(train_log["action_names"])
    d = train_log["contexts"].shape[1]
    print("Loaded IST. K =", K, "| d =", d)

    n_steps = 200
    seed = 0

    # algorithms to run (each  returns a function that builds a fresh algo per site)
    algo_specs = [
        ("RandomPolicy", lambda site_id: RandomPolicy(n_actions=K, seed=seed + site_id)),
        ("LinUCB(a=1.0)", lambda site_id: LinUCB(n_actions=K, context_dim=d, alpha=1.0)),
        ("LinearTS(lam=1,sigma=1)", lambda site_id: LinearTS(n_actions=K, context_dim=d, lam=1.0, sigma=1.0, seed=seed + site_id)),
    ]

    for algo_name, make_algo in algo_specs:
        # TRAIN
        train_rewards, train_acc = run_phase("TRAIN", train_log, train_sites, make_algo, n_steps, seed)
        # TEST
        test_rewards, test_acc = run_phase("TEST", test_log, test_sites, make_algo, n_steps, seed)

        summarize(train_rewards, f"TRAIN per-site mean reward ({algo_name})")
        summarize(test_rewards, f"TEST  per-site mean reward ({algo_name})")

        summarize(train_acc, f"TRAIN acceptance rates ({algo_name})")
        summarize(test_acc, f"TEST  acceptance rates ({algo_name})")


if __name__ == "__main__":
    main()