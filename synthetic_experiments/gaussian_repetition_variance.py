"""
Repetition-level pipeline for strict 2-arm Gaussian bandits.

Workflow per repetition:
1) Sample offline data from true env with uniform random policy.
2) Build rejection simulator and plugin simulator.
3) Tune UCB alpha on each simulator.
4) Run tuned UCB on true env and record mean reward.

Outputs:
- repetition-level CSV with one row per (repetition, simulator approach)
- optional distribution plots via evaluation.visualization
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.ucb import UCB
from data.gen_data import RandomPolicy, rollout_dataset
from envs.data_driven_env import build_simulator
from envs.gaussian_two_arm_env import GaussianTwoArmEnv
from evaluation.runner import EvaluatorRunner
from evaluation.visualization import plot_mean_reward_distribution, plot_running_mean_ci_from_csv
from simulator_generation.gaussian_plugin_generator import GaussianPluginGenerator


def _get_field(ds, candidates):
    for k in candidates:
        if hasattr(ds, k):
            return np.asarray(getattr(ds, k))
        if isinstance(ds, dict) and k in ds:
            return np.asarray(ds[k])
    raise KeyError(f"Could not find any of fields {candidates} in offline_dataset.")

def estimate_shared_sigma_hat(offline_dataset, n_actions=2, eps=1e-8) -> float:
    r = _get_field(offline_dataset, ["rewards", "reward", "r", "R", "y"]).astype(float)
    a = _get_field(offline_dataset, ["actions", "action", "a", "A"]).astype(int)

    vars_, counts = [], []
    for arm in range(n_actions):
        rr = r[a == arm]
        if rr.size >= 2:
            vars_.append(np.var(rr, ddof=1))
            counts.append(rr.size)

    if len(vars_) == n_actions:
        sigma2_hat = float(np.average(vars_, weights=counts))  
    else:
        sigma2_hat = float(np.var(r, ddof=1)) if r.size >= 2 else 0.0  

    return float(np.sqrt(max(sigma2_hat, eps)))



def _mean_reward_from_runner_metrics(metrics: Dict[str, np.ndarray]) -> float:
    valid_steps = int(metrics["valid_steps"][0]) if "valid_steps" in metrics else 0
    cumulative_reward = metrics.get("cumulative_reward", np.array([]))
    if valid_steps <= 0 or len(cumulative_reward) == 0:
        return float("nan")
    return float(cumulative_reward[-1] / valid_steps)


def _evaluate_ucb_once(env, alpha: float, sigma: float, n_steps: int) -> Tuple[float, int]:
    algo = UCB(n_actions=2, alpha=alpha, sigma=sigma)
    runner = EvaluatorRunner(env=env, algo=algo, n_steps=n_steps)
    metrics = runner.run_experiment()
    mean_reward = _mean_reward_from_runner_metrics(metrics)
    valid_steps = int(metrics["valid_steps"][0]) if "valid_steps" in metrics else 0
    return mean_reward, valid_steps


def _tune_alpha(
    simulator_builder,
    alpha_grid: List[float],
    sigma_for_algo: float,
    offline_eval_steps: int,
    n_trials: int,
) -> Tuple[float, float]:
    best_alpha = alpha_grid[0]
    best_avg_reward = -np.inf

    for alpha in alpha_grid:
        trial_rewards: List[float] = []
        for _ in range(n_trials):
            simulator = simulator_builder()
            mean_reward, _ = _evaluate_ucb_once(
                env=simulator,
                alpha=alpha,
                sigma=sigma_for_algo,
                n_steps=offline_eval_steps,
            )
            if not np.isnan(mean_reward):
                trial_rewards.append(mean_reward)

        avg_reward = float(np.mean(trial_rewards)) if trial_rewards else -np.inf
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_alpha = alpha

    return best_alpha, best_avg_reward


def _validate_args(args: argparse.Namespace) -> None:
    if args.repetitions <= 0:
        raise ValueError("repetitions must be > 0")
    if args.offline_data_steps <= 0:
        raise ValueError("offline_data_steps must be > 0")
    if args.offline_eval_steps <= 0:
        raise ValueError("offline_eval_steps must be > 0")
    if args.online_eval_steps <= 0:
        raise ValueError("online_eval_steps must be > 0")
    if args.offline_trials <= 0:
        raise ValueError("offline_trials must be > 0")
    if args.offline_data_steps < args.offline_eval_steps:
        raise ValueError("offline_data_steps must be >= offline_eval_steps for stable tuning")
    if not args.alpha_grid:
        raise ValueError("alpha_grid must be non-empty")
    if any(alpha <= 0 for alpha in args.alpha_grid):
        raise ValueError("all alpha values must be > 0")
    if args.sigma <= 0:
        raise ValueError("sigma must be > 0")


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    _validate_args(args)
    rng = np.random.default_rng(args.seed)
    alpha_grid = [float(x) for x in args.alpha_grid]

    rows = []

    for rep in range(args.repetitions):
        rep_seed = int(rng.integers(1, 10_000_000))

        true_env_for_data = GaussianTwoArmEnv(delta=args.delta, sigma=args.sigma, seed=rep_seed)
        random_policy = RandomPolicy(n_actions=2, seed=rep_seed + 11)
        offline_dataset = rollout_dataset(
            env=true_env_for_data,
            behavior_policy=random_policy,
            n_steps=args.offline_data_steps,
        )

        plugin_generator = GaussianPluginGenerator(seed=rep_seed + 29)
        plugin_generator.fit(offline_dataset)
        sigma_true = float(args.sigma)
        sigma_hat = estimate_shared_sigma_hat(offline_dataset, n_actions=2)
        sigma_ratio = sigma_hat / sigma_true
        underestimate = int(sigma_hat < sigma_true)

        rejection_builder = lambda: build_simulator(offline_dataset)
        plugin_builder = lambda: plugin_generator.generate_env(
            {"seed": int(rng.integers(1, 10_000_000))}
        )

        rej_alpha, rej_offline_mean = _tune_alpha(
            simulator_builder=rejection_builder,
            alpha_grid=alpha_grid,
            sigma_for_algo=args.sigma,
            offline_eval_steps=args.offline_eval_steps,
            n_trials=args.offline_trials,
        )

        plugin_alpha, plugin_offline_mean = _tune_alpha(
            simulator_builder=plugin_builder,
            alpha_grid=alpha_grid,
            sigma_for_algo=args.sigma,
            offline_eval_steps=args.offline_eval_steps,
            n_trials=args.offline_trials,
        )

        true_env_rej = GaussianTwoArmEnv(delta=args.delta, sigma=sigma_hat, seed=rep_seed + 101)
        rej_online_mean, rej_online_valid = _evaluate_ucb_once(
            env=true_env_rej,
            alpha=rej_alpha,
            sigma=args.sigma,
            n_steps=args.online_eval_steps,
        )

        true_env_plugin = GaussianTwoArmEnv(delta=args.delta,sigma=sigma_hat, seed=rep_seed + 101)
        plugin_online_mean, plugin_online_valid = _evaluate_ucb_once(
            env=true_env_plugin,
            alpha=plugin_alpha,
            sigma=args.sigma,
            n_steps=args.online_eval_steps,
        )

        rows.append(
            {
                "repetition": rep,
                "approach": "rejection",
                "best_alpha": rej_alpha,
                "offline_mean_reward": rej_offline_mean,
                "online_mean_reward": rej_online_mean,
                "offline_data_steps": args.offline_data_steps,
                "offline_eval_steps": args.offline_eval_steps,
                "online_eval_steps": args.online_eval_steps,
                "online_valid_steps": rej_online_valid,
                "delta": args.delta,
                "sigma": args.sigma,
                "sigma_true": sigma_true,
                "sigma_hat": sigma_hat,
                "sigma_ratio": sigma_ratio,
                "underestimate": underestimate,
            }
        )

        rows.append(
            {
                "repetition": rep,
                "approach": "plugin",
                "best_alpha": plugin_alpha,
                "offline_mean_reward": plugin_offline_mean,
                "online_mean_reward": plugin_online_mean,
                "offline_data_steps": args.offline_data_steps,
                "offline_eval_steps": args.offline_eval_steps,
                "online_eval_steps": args.online_eval_steps,
                "online_valid_steps": plugin_online_valid,
                "delta": args.delta,
                "sigma": args.sigma,
                "sigma_true": sigma_true,
                "sigma_hat": sigma_hat,
                "sigma_ratio": sigma_ratio,
                "underestimate": underestimate,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian 2-arm repetition pipeline")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--offline_data_steps", type=int, default=5000)
    parser.add_argument("--offline_eval_steps", type=int, default=1000)
    parser.add_argument("--online_eval_steps", type=int, default=1000)
    parser.add_argument("--offline_trials", type=int, default=3)
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_output", type=str, default="logs/gaussian_repetition_results.csv")
    parser.add_argument("--plot_output", type=str, default="logs/gaussian_repetition_mean_reward.png")
    parser.add_argument("--plot_running_ci_output", type=str, default="")
    parser.add_argument("--no_show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_pipeline(args)

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    plot_mean_reward_distribution(
        csv_path=str(csv_path),
        value_col="online_mean_reward",
        group_col="approach",
        output_path=args.plot_output,
        show=not args.no_show,
        title="Online Mean Reward Distribution Across Repetitions",
    )

    running_ci_output = args.plot_running_ci_output
    if not running_ci_output:
        base = Path(args.plot_output)
        stem = base.stem.replace("mean_reward", "running_mean_ci")
        if stem == base.stem:
            stem = f"{base.stem}_running_mean_ci"
        running_ci_output = str(base.with_name(f"{stem}{base.suffix}"))

    plot_running_mean_ci_from_csv(
        csv_path=str(csv_path),
        value_col="online_mean_reward",
        group_col="approach",
        repetition_col="repetition",
        output_path=running_ci_output,
        show=not args.no_show,
        title="Running Mean ± 95% CI of Online Mean Reward",
    )

    summary = df.groupby("approach")["online_mean_reward"].agg(["mean", "std", "count"])
    print("Saved repetition-level results:", csv_path)
    print(summary.to_string())


if __name__ == "__main__":
    main()
