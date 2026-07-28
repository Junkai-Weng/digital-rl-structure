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
import platform
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


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


def _parse_alpha_grid(alpha_grid_arg) -> List[float]:
    if isinstance(alpha_grid_arg, np.ndarray):
        values = np.asarray(alpha_grid_arg, dtype=float).ravel()
    elif isinstance(alpha_grid_arg, (list, tuple)):
        if len(alpha_grid_arg) == 1 and isinstance(alpha_grid_arg[0], str):
            return _parse_alpha_grid(alpha_grid_arg[0])
        try:
            values = np.asarray(alpha_grid_arg, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha_grid list must contain numeric values") from exc
    elif isinstance(alpha_grid_arg, str):
        text = alpha_grid_arg.strip()
        if not text:
            raise ValueError("alpha_grid must be non-empty")

        safe_globals = {"__builtins__": {}}
        safe_locals = {
            "np": np,
            "logspace": np.logspace,
            "linspace": np.linspace,
            "geomspace": np.geomspace,
            "array": np.array,
        }

        parsed = None
        try:
            parsed = eval(text, safe_globals, safe_locals)
        except Exception:
            cleaned = text.strip("[]")
            tokens = [tok for tok in cleaned.replace(",", " ").split() if tok]
            if not tokens:
                raise ValueError("alpha_grid must be non-empty")
            try:
                parsed = [float(tok) for tok in tokens]
            except ValueError as exc:
                raise ValueError(
                    "alpha_grid must be numeric list or numpy expression like np.logspace(-1, 1, 20)"
                ) from exc

        try:
            values = np.asarray(parsed, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha_grid expression must evaluate to numeric values") from exc
    else:
        raise TypeError("alpha_grid must be list, numpy array, or string expression")

    if values.size == 0:
        raise ValueError("alpha_grid must be non-empty")

    grid = [float(x) for x in values.tolist()]
    if any(alpha <= 0 for alpha in grid):
        raise ValueError("all alpha values must be > 0")
    return grid


def _evaluate_ucb_once(env, alpha: float, n_steps: int, sigma_scale: float = 1.0) -> Tuple[float, int]:
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be > 0")
    algo = UCB(n_actions=2, alpha=float(alpha) * float(sigma_scale))
    runner = EvaluatorRunner(env=env, algo=algo, n_steps=n_steps)
    metrics = runner.run_experiment()
    mean_reward = _mean_reward_from_runner_metrics(metrics)
    valid_steps = int(metrics["valid_steps"][0]) if "valid_steps" in metrics else 0
    return mean_reward, valid_steps


def _tune_alpha(
    simulator_builder,
    alpha_grid: List[float],
    offline_eval_steps: int,
    n_trials: int,
    sigma_scale: float = 1.0,
    reuse_simulator: bool = False,
) -> Tuple[float, float]:
    best_alpha = alpha_grid[0]
    best_avg_reward = -np.inf

    simulator = simulator_builder() if reuse_simulator else None

    for alpha in alpha_grid:
        trial_rewards: List[float] = []
        for _ in range(n_trials):
            trial_env = simulator if reuse_simulator else simulator_builder()
            mean_reward, _ = _evaluate_ucb_once(
                env=trial_env,
                alpha=alpha,
                n_steps=offline_eval_steps,
                sigma_scale=sigma_scale,
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
    if args.sigma <= 0:
        raise ValueError("sigma must be > 0")


def _resolve_n_jobs(n_jobs: int) -> int:
    cpu_total = max(int(cpu_count()), 1)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if n_jobs < 0:
        return max(1, cpu_total + 1 + n_jobs)
    return max(1, n_jobs)


def _resolve_joblib_temp_folder() -> str | None:
    if platform.system().lower() != "windows":
        return None

    candidates = [
        PROJECT_ROOT / ".joblib_tmp",
        Path("C:/joblib_tmp"),
    ]
    for candidate in candidates:
        text = str(candidate)
        try:
            text.encode("ascii")
        except UnicodeEncodeError:
            continue
        candidate.mkdir(parents=True, exist_ok=True)
        return text
    return None


def run_single_repetition(rep: int, args: argparse.Namespace, alpha_grid: List[float]) -> List[Dict[str, float]]:
    rep_rng = np.random.default_rng(args.seed + rep)
    rep_seed = int(rep_rng.integers(1, 10_000_000))

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
        {"seed": int(rep_rng.integers(1, 10_000_000))}
    )

    rej_true_alpha, rej_true_offline_mean = _tune_alpha(
        simulator_builder=rejection_builder,
        alpha_grid=alpha_grid,
        offline_eval_steps=args.offline_eval_steps,
        n_trials=args.offline_trials,
        sigma_scale=sigma_true,
        reuse_simulator=True,
    )

    rej_hat_alpha, rej_hat_offline_mean = _tune_alpha(
        simulator_builder=rejection_builder,
        alpha_grid=alpha_grid,
        offline_eval_steps=args.offline_eval_steps,
        n_trials=args.offline_trials,
        sigma_scale=sigma_hat,
        reuse_simulator=True,
    )

    plugin_true_alpha, plugin_true_offline_mean = _tune_alpha(
        simulator_builder=plugin_builder,
        alpha_grid=alpha_grid,
        offline_eval_steps=args.offline_eval_steps,
        n_trials=args.offline_trials,
        sigma_scale=sigma_true,
        reuse_simulator=False,
    )

    plugin_hat_alpha, plugin_hat_offline_mean = _tune_alpha(
        simulator_builder=plugin_builder,
        alpha_grid=alpha_grid,
        offline_eval_steps=args.offline_eval_steps,
        n_trials=args.offline_trials,
        sigma_scale=sigma_hat,
        reuse_simulator=False,
    )

    true_env_rej_true = GaussianTwoArmEnv(delta=args.delta, sigma=args.sigma, seed=rep_seed + 101)
    rej_true_online_mean, rej_true_online_valid = _evaluate_ucb_once(
        env=true_env_rej_true,
        alpha=rej_true_alpha,
        n_steps=args.online_eval_steps,
        sigma_scale=sigma_true,
    )

    true_env_rej_hat = GaussianTwoArmEnv(delta=args.delta, sigma=args.sigma, seed=rep_seed + 101)
    rej_hat_online_mean, rej_hat_online_valid = _evaluate_ucb_once(
        env=true_env_rej_hat,
        alpha=rej_hat_alpha,
        n_steps=args.online_eval_steps,
        sigma_scale=sigma_hat,
    )

    true_env_plugin_true = GaussianTwoArmEnv(delta=args.delta, sigma=args.sigma, seed=rep_seed + 101)
    plugin_true_online_mean, plugin_true_online_valid = _evaluate_ucb_once(
        env=true_env_plugin_true,
        alpha=plugin_true_alpha,
        n_steps=args.online_eval_steps,
        sigma_scale=sigma_true,
    )

    true_env_plugin_hat = GaussianTwoArmEnv(delta=args.delta, sigma=args.sigma, seed=rep_seed + 101)
    plugin_hat_online_mean, plugin_hat_online_valid = _evaluate_ucb_once(
        env=true_env_plugin_hat,
        alpha=plugin_hat_alpha,
        n_steps=args.online_eval_steps,
        sigma_scale=sigma_hat,
    )

    rows: List[Dict[str, float]] = []
    rows.append(
        {
            "repetition": rep,
            "approach": "rejection+true",
            "best_alpha": rej_true_alpha,
            "offline_mean_reward": rej_true_offline_mean,
            "online_mean_reward": rej_true_online_mean,
            "offline_data_steps": args.offline_data_steps,
            "offline_eval_steps": args.offline_eval_steps,
            "online_eval_steps": args.online_eval_steps,
            "online_valid_steps": rej_true_online_valid,
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
            "approach": "rejection+hat",
            "best_alpha": rej_hat_alpha,
            "offline_mean_reward": rej_hat_offline_mean,
            "online_mean_reward": rej_hat_online_mean,
            "offline_data_steps": args.offline_data_steps,
            "offline_eval_steps": args.offline_eval_steps,
            "online_eval_steps": args.online_eval_steps,
            "online_valid_steps": rej_hat_online_valid,
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
            "approach": "plugin+true",
            "best_alpha": plugin_true_alpha,
            "offline_mean_reward": plugin_true_offline_mean,
            "online_mean_reward": plugin_true_online_mean,
            "offline_data_steps": args.offline_data_steps,
            "offline_eval_steps": args.offline_eval_steps,
            "online_eval_steps": args.online_eval_steps,
            "online_valid_steps": plugin_true_online_valid,
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
            "approach": "plugin+hat",
            "best_alpha": plugin_hat_alpha,
            "offline_mean_reward": plugin_hat_offline_mean,
            "online_mean_reward": plugin_hat_online_mean,
            "offline_data_steps": args.offline_data_steps,
            "offline_eval_steps": args.offline_eval_steps,
            "online_eval_steps": args.online_eval_steps,
            "online_valid_steps": plugin_hat_online_valid,
            "delta": args.delta,
            "sigma": args.sigma,
            "sigma_true": sigma_true,
            "sigma_hat": sigma_hat,
            "sigma_ratio": sigma_ratio,
            "underestimate": underestimate,
        }
    )
    return rows


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    _validate_args(args)
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    n_jobs = _resolve_n_jobs(args.n_jobs)

    if n_jobs == 1 or args.repetitions == 1:
        rep_iter = range(args.repetitions)
        if tqdm is not None:
            rep_iter = tqdm(rep_iter, total=args.repetitions, desc="repetitions")
        results = [run_single_repetition(rep, args, alpha_grid) for rep in rep_iter]
    else:
        print(f"Starting {args.repetitions} repetitions using {n_jobs} processes (joblib)...")
        rep_iter = range(args.repetitions)
        if tqdm is not None:
            rep_iter = tqdm(rep_iter, total=args.repetitions, desc="repetitions")
        joblib_temp_folder = _resolve_joblib_temp_folder()
        results = Parallel(n_jobs=n_jobs, prefer="processes", temp_folder=joblib_temp_folder)(
            delayed(run_single_repetition)(rep, args, alpha_grid)
            for rep in rep_iter
        )

    flat_rows = [row for rep_rows in results for row in rep_rows]
    return pd.DataFrame(flat_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian 2-arm repetition pipeline")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--offline_data_steps", type=int, default=5000)
    parser.add_argument("--offline_eval_steps", type=int, default=1000)
    parser.add_argument("--online_eval_steps", type=int, default=1000)
    parser.add_argument("--offline_trials", type=int, default=3)
    parser.add_argument("--alpha_grid", type=str, nargs="+", default=["0.25", "0.5", "1.0", "2.0", "4.0"])
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of worker processes for repetitions; -1 uses all CPUs, 1 disables multiprocessing.",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="logs/gaussian_repe_plug_rej/gaussian_repetition_results.csv",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default="logs/gaussian_repe_plug_rej/gaussian_repetition_mean_reward.png",
    )
    parser.add_argument("--plot_running_ci_output", type=str, default="")
    parser.add_argument("--no_show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_pipeline(args)

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    plot_path = Path(args.plot_output)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

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

    running_ci_path = Path(running_ci_output)
    running_ci_path.parent.mkdir(parents=True, exist_ok=True)

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
    global_best_beta = (
        df.groupby(["approach", "best_alpha"], as_index=False)["online_mean_reward"]
        .mean()
        .sort_values(["approach", "online_mean_reward"], ascending=[True, False])
        .groupby("approach", as_index=False)
        .first()
    )

    print("Saved repetition-level results:", csv_path)
    print(
        "Estimated sigma_hat formula: sigma_hat = sqrt(weighted average of per-arm sample variances); "
        "fallback to sqrt(global sample variance) if any arm has fewer than 2 samples."
    )
    print("Global best beta (by highest mean online reward):")
    print(global_best_beta.to_string(index=False))
    print(summary.to_string())


if __name__ == "__main__":
    main()
