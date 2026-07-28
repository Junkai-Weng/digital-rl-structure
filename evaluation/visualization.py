"""
Visualization utilities for evaluator logs.

This module reads logs saved by `EvaluatorRunner.run_experiment(log_path=...)`
and plots cumulative reward and cumulative regret.

It supports:
1) single-log plotting (backward compatible),
2) multi-policy comparison in one figure with optional mean±std bands.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _derive_regret_output_path(output_path: Optional[str]) -> Optional[Path]:
	"""Derive output filename for cumulative regret figure.

	Examples:
	- xxx_cum_reward.png -> xxx_cum_regret.png
	- xxx.png -> xxx_cum_regret.png
	"""
	if output_path is None:
		return None

	reward_path = Path(output_path)
	stem = reward_path.stem
	if "reward" in stem:
		regret_stem = stem.replace("reward", "regret")
	else:
		regret_stem = f"{stem}_cum_regret"

	return reward_path.with_name(f"{regret_stem}{reward_path.suffix}")


def _derive_compare_output_path(output_path: Optional[str], metric_name: str) -> Optional[Path]:
	"""Derive compare figure output filename for a given metric.

	Examples:
	- compare.png + cumulative_reward -> compare_cumulative_reward.png
	- compare_cumulative_reward.png + cumulative_regret -> compare_cumulative_regret.png
	"""
	if output_path is None:
		return None

	base = Path(output_path)
	stem = base.stem
	if "cumulative_reward" in stem or "cumulative_regret" in stem:
		new_stem = stem.replace("cumulative_reward", metric_name).replace("cumulative_regret", metric_name)
	else:
		new_stem = f"{stem}_{metric_name}"

	return base.with_name(f"{new_stem}{base.suffix}")


def load_runner_logs(log_path: str) -> dict:
	"""Load all arrays from a runner-produced .npz log file.

	Returns a plain dictionary mapping metric name -> numpy array.
	"""
	path = Path(log_path)
	if not path.exists():
		# Improve error discoverability by listing sibling .npz files.
		parent = path.parent if str(path.parent) else Path(".")
		available = []
		if parent.exists():
			available = sorted(p.name for p in parent.glob("*.npz"))
		message = f"Log file not found: {path}"
		if available:
			message += f". Available .npz files in '{parent}': {', '.join(available)}"
		raise FileNotFoundError(message)

	with np.load(path) as data:
		return {key: data[key] for key in data.files}


def _parse_policy_log_specs(specs: List[str]) -> Dict[str, List[str]]:
	"""Parse CLI specs like `LinUCB=logs/linucb_seed*.npz`.

	Returns mapping: policy label -> resolved list of file paths.
	"""
	parsed: Dict[str, List[str]] = {}
	for spec in specs:
		if "=" not in spec:
			raise ValueError(f"Invalid --policy_logs spec '{spec}'. Expected format: Label=path_or_glob")

		label, raw_pattern = spec.split("=", 1)
		label = label.strip()
		raw_pattern = raw_pattern.strip()
		if not label or not raw_pattern:
			raise ValueError(f"Invalid --policy_logs spec '{spec}'. Label and path pattern cannot be empty")

		matches = sorted(glob.glob(raw_pattern))
		if not matches:
			raise FileNotFoundError(f"No log files matched for policy '{label}' with pattern: {raw_pattern}")

		if label in parsed:
			parsed[label].extend(matches)
		else:
			parsed[label] = matches

	# Deduplicate while preserving order
	for key in parsed:
		seen = set()
		unique = []
		for path in parsed[key]:
			if path not in seen:
				seen.add(path)
				unique.append(path)
		parsed[key] = unique

	return parsed


def _collect_metric_runs(log_paths: List[str], metric_key: str) -> Tuple[np.ndarray, int]:
	"""Load metric arrays from multiple logs and return aligned stack + min length.

	All runs are truncated to the shortest length to align x-axis.
	"""
	runs: List[np.ndarray] = []
	for path in log_paths:
		logs = load_runner_logs(path)
		metric = logs.get(metric_key)
		if metric is None or len(metric) == 0:
			continue
		runs.append(np.asarray(metric, dtype=float))

	if not runs:
		raise ValueError(f"No valid '{metric_key}' arrays found in logs: {log_paths}")

	min_len = min(len(run) for run in runs)
	stack = np.stack([run[:min_len] for run in runs], axis=0)
	return stack, min_len


def plot_policy_comparison(
	policy_logs: Dict[str, List[str]],
	metric_key: str = "cumulative_reward",
	output_path: Optional[str] = None,
	show: bool = True,
	show_std_band: bool = True,
	title: Optional[str] = None,
) -> None:
	"""Compare multiple policies in one figure with mean ± std band.

	Args:
		policy_logs: Mapping policy label -> list of log files.
		metric_key: One of {'cumulative_reward', 'cumulative_regret'}.
		output_path: Optional output file path.
		show: Whether to display interactively.
		show_std_band: Draw standard deviation band when >=2 runs per policy.
		title: Optional custom figure title.
	"""
	if metric_key not in {"cumulative_reward", "cumulative_regret"}:
		raise ValueError("metric_key must be 'cumulative_reward' or 'cumulative_regret'")

	try:
		plt.style.use("seaborn-v0_8-whitegrid")
	except OSError:
		# fallback to default matplotlib style when seaborn style is unavailable
		pass

	plt.figure(figsize=(9, 5.5))

	for label, paths in policy_logs.items():
		stack, n_steps = _collect_metric_runs(paths, metric_key)
		mean_curve = np.mean(stack, axis=0)
		std_curve = np.std(stack, axis=0)
		steps = np.arange(1, n_steps + 1)

		line = plt.plot(steps, mean_curve, linewidth=2.2, label=f"{label} (n={stack.shape[0]})")[0]

		if show_std_band and stack.shape[0] > 1:
			plt.fill_between(
				steps,
				mean_curve - std_curve,
				mean_curve + std_curve,
				alpha=0.20,
				color=line.get_color(),
			)

	plot_title = title or f"Policy Comparison: {metric_key.replace('_', ' ').title()}"
	plt.title(plot_title)
	plt.xlabel("Valid Step")
	plt.ylabel(metric_key.replace("_", " ").title())
	plt.legend()
	plt.tight_layout()

	if output_path:
		output = Path(output_path)
		output.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output, dpi=220)

	if show:
		plt.show()
	else:
		plt.close()


def plot_cumulative_reward(log_path: str, output_path: Optional[str] = None, show: bool = True) -> None:
	"""Plot cumulative reward and, when available, cumulative regret from runner logs.

	The function always attempts to plot reward first. If the log also contains
	`cumulative_regret`, it generates an additional regret figure.
	"""
	# 1) Read metrics arrays from the serialized .npz log file.
	logs = load_runner_logs(log_path)
	cumulative_reward = logs.get("cumulative_reward")
	cumulative_regret = logs.get("cumulative_regret")

	# Reward is mandatory for this visualization entrypoint.
	if cumulative_reward is None or len(cumulative_reward) == 0:
		raise ValueError("The log does not contain a non-empty 'cumulative_reward' array.")

	# Use 1-based step index to match experiment reporting style.
	steps = np.arange(1, len(cumulative_reward) + 1)

	# 2) Draw cumulative reward curve.
	plt.figure(figsize=(8, 5))
	plt.plot(steps, cumulative_reward, label="Cumulative Reward", linewidth=2)
	plt.xlabel("Valid Step")
	plt.ylabel("Cumulative Reward")
	plt.title("Cumulative Reward Curve")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()

	if output_path:
		# Create parent folder if needed and save reward figure.
		output = Path(output_path)
		output.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output, dpi=200)

	if show:
		plt.show()
	else:
		plt.close()

	if cumulative_regret is None or len(cumulative_regret) == 0:
		# Regret is optional; skip gracefully when not present.
		print("[visualization] 'cumulative_regret' not found in logs, skipped regret plot.")
		return

	# 3) Draw cumulative regret curve when available.
	regret_steps = np.arange(1, len(cumulative_regret) + 1)
	regret_output = _derive_regret_output_path(output_path)

	plt.figure(figsize=(8, 5))
	plt.plot(regret_steps, cumulative_regret, label="Cumulative Regret", linewidth=2)
	plt.xlabel("Valid Step")
	plt.ylabel("Cumulative Regret")
	plt.title("Cumulative Regret Curve")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()

	if regret_output:
		# Save regret figure with an auto-derived filename.
		regret_output.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(regret_output, dpi=200)

	if show:
		plt.show()
	else:
		plt.close()


def plot_mean_reward_distribution(
	csv_path: str,
	value_col: str = "online_mean_reward",
	group_col: str = "approach",
	output_path: Optional[str] = None,
	show: bool = True,
	title: Optional[str] = None,
) -> None:
	"""Plot repetition-level mean reward distribution from a CSV file.

	Expected CSV format: one row per repetition with at least
	`value_col` and `group_col`.
	"""
	data = pd.read_csv(csv_path)
	if value_col not in data.columns:
		raise ValueError(f"Column not found: {value_col}")
	if group_col not in data.columns:
		raise ValueError(f"Column not found: {group_col}")

	plot_df = data[[group_col, value_col]].dropna().copy()
	if len(plot_df) == 0:
		raise ValueError("No valid rows to plot after dropping NaN values")

	groups = list(plot_df[group_col].unique())
	series = [plot_df.loc[plot_df[group_col] == g, value_col].to_numpy() for g in groups]

	try:
		plt.style.use("seaborn-v0_8-whitegrid")
	except OSError:
		pass

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

	axes[0].boxplot(series, labels=groups, patch_artist=True)
	axes[0].set_title("Boxplot")
	axes[0].set_xlabel(group_col)
	axes[0].set_ylabel(value_col)

	for idx, g in enumerate(groups):
		x = series[idx]
		hist_label = f"{g} (n={len(x)})"
		axes[1].hist(x, bins=20, alpha=0.5, label=hist_label)
	axes[1].set_title("Histogram")
	axes[1].set_xlabel(value_col)
	axes[1].set_ylabel("Count")
	axes[1].legend()

	fig.suptitle(title or "Mean Reward Distribution Across Repetitions")
	fig.tight_layout()

	if output_path:
		output = Path(output_path)
		output.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output, dpi=220)

	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_running_mean_ci_from_csv(
	csv_path: str,
	value_col: str = "online_mean_reward",
	group_col: str = "approach",
	repetition_col: str = "repetition",
	output_path: Optional[str] = None,
	show: bool = True,
	title: Optional[str] = None,
	confidence_z: float = 1.96,
) -> None:
	"""Plot running mean and 95% CI by repetition order for each approach.

	For each group, rows are sorted by `repetition_col`; at prefix n we plot:
	- running mean of `value_col`
	- confidence band mean ± z * std / sqrt(n)
	"""
	data = pd.read_csv(csv_path)
	for required_col in (value_col, group_col, repetition_col):
		if required_col not in data.columns:
			raise ValueError(f"Column not found: {required_col}")

	plot_df = data[[group_col, repetition_col, value_col]].dropna().copy()
	if len(plot_df) == 0:
		raise ValueError("No valid rows to plot after dropping NaN values")

	try:
		plt.style.use("seaborn-v0_8-whitegrid")
	except OSError:
		pass

	plt.figure(figsize=(9, 5.4))

	groups = list(plot_df[group_col].unique())
	for group_name in groups:
		group_df = plot_df.loc[plot_df[group_col] == group_name].sort_values(by=repetition_col)
		x = group_df[value_col].to_numpy(dtype=float)
		if len(x) == 0:
			continue

		n = np.arange(1, len(x) + 1)
		running_mean = np.cumsum(x) / n

		running_std = np.zeros_like(x, dtype=float)
		for idx in range(len(x)):
			prefix = x[: idx + 1]
			if len(prefix) >= 2:
				running_std[idx] = float(np.std(prefix, ddof=1))

		stderr = np.divide(running_std, np.sqrt(n), out=np.zeros_like(running_std), where=n > 0)
		ci = confidence_z * stderr

		line = plt.plot(n, running_mean, linewidth=2.2, label=f"{group_name}")[0]
		plt.fill_between(n, running_mean - ci, running_mean + ci, alpha=0.20, color=line.get_color())

	plt.xlabel("Repetition Prefix Length")
	plt.ylabel(value_col)
	plt.title(title or "Running Mean ± 95% CI Across Repetitions")
	plt.legend()
	plt.tight_layout()

	if output_path:
		output = Path(output_path)
		output.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output, dpi=220)

	if show:
		plt.show()
	else:
		plt.close()


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for reward/regret visualization."""
	parser = argparse.ArgumentParser(description="Visualize runner logs: single-log or multi-policy comparison")
	parser.add_argument("--log_path", type=str, default=None, help="Path to one runner .npz log file (single-log mode)")
	parser.add_argument(
		"--policy_logs",
		action="append",
		default=[],
		help="Comparison mode input, repeatable: Label=path_or_glob. Example: --policy_logs LinUCB=logs/linucb_seed*.npz",
	)
	parser.add_argument(
		"--metrics",
		nargs="+",
		default=["cumulative_reward"],
		choices=["cumulative_reward", "cumulative_regret"],
		help="Metrics to plot in comparison mode",
	)
	parser.add_argument("--title", type=str, default=None, help="Optional custom title for comparison figure")
	parser.add_argument("--no_std_band", action="store_true", help="Disable standard deviation band in comparison mode")
	parser.add_argument("--output_path", type=str, default=None, help="Optional path to save the figure")
	parser.add_argument("--no_show", action="store_true", help="Do not open interactive plot window")
	parser.add_argument("--label", type=str, default="Run", help="Legend label for single-log mode")
	return parser.parse_args()


def main() -> None:
	"""CLI entrypoint."""
	args = parse_args()

	if args.policy_logs:
		policy_logs = _parse_policy_log_specs(args.policy_logs)
		for metric_name in args.metrics:
			metric_output = _derive_compare_output_path(args.output_path, metric_name)
			plot_policy_comparison(
				policy_logs=policy_logs,
				metric_key=metric_name,
				output_path=str(metric_output) if metric_output is not None else None,
				show=not args.no_show,
				show_std_band=not args.no_std_band,
				title=args.title,
			)
		return

	if not args.log_path:
		raise ValueError("Either --log_path (single mode) or --policy_logs (comparison mode) must be provided")

	# Backward-compatible behavior for single-log mode.
	logs = load_runner_logs(args.log_path)
	cumulative_reward = logs.get("cumulative_reward")
	if cumulative_reward is None or len(cumulative_reward) == 0:
		raise ValueError("The log does not contain a non-empty 'cumulative_reward' array.")

	steps = np.arange(1, len(cumulative_reward) + 1)
	try:
		plt.style.use("seaborn-v0_8-whitegrid")
	except OSError:
		pass

	plt.figure(figsize=(8, 5))
	plt.plot(steps, cumulative_reward, label=f"{args.label}: Cumulative Reward", linewidth=2.2)
	plt.xlabel("Valid Step")
	plt.ylabel("Cumulative Reward")
	plt.title("Cumulative Reward Curve")
	plt.legend()
	plt.tight_layout()

	if args.output_path:
		output = Path(args.output_path)
		output.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output, dpi=220)

	if not args.no_show:
		plt.show()
	else:
		plt.close()

	cumulative_regret = logs.get("cumulative_regret")
	if cumulative_regret is None or len(cumulative_regret) == 0:
		print("[visualization] 'cumulative_regret' not found in logs, skipped regret plot.")
		return

	regret_steps = np.arange(1, len(cumulative_regret) + 1)
	regret_output = _derive_regret_output_path(args.output_path)
	plt.figure(figsize=(8, 5))
	plt.plot(regret_steps, cumulative_regret, label=f"{args.label}: Cumulative Regret", linewidth=2.2)
	plt.xlabel("Valid Step")
	plt.ylabel("Cumulative Regret")
	plt.title("Cumulative Regret Curve")
	plt.legend()
	plt.tight_layout()

	if regret_output:
		regret_output.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(regret_output, dpi=220)

	if not args.no_show:
		plt.show()
	else:
		plt.close()


if __name__ == "__main__":
	main()
