"""CLI for generating offline contextual-bandit datasets.

This script rolls out a configurable behavior policy in a configurable
environment, writes a unified offline dataset, and can optionally:
1) create train/test splits,
2) serialize a data-driven simulator for later reuse.
"""

import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.base_algo import BaseBanditAlgorithm
from algorithms.lints import LinTS
from algorithms.linucb import LinUCB
from data.dataset import OfflineBanditDataset
from envs.base_env import BaseBanditEnv
from envs.data_driven_env import build_simulator


class RandomPolicy(BaseBanditAlgorithm):
	"""Non-learning uniform random behavior policy for data collection."""

	def __init__(self, n_actions: int, seed: int = 42):
		super().__init__(n_actions=n_actions)
		self.rng = np.random.default_rng(seed)

	def select_action(self, context: np.ndarray) -> int:
		return int(self.rng.integers(self.n_actions))

	def update(self, context: np.ndarray, action: int, reward: float, info: Optional[Dict[str, Any]] = None) -> None:
		return None


def _load_class(class_path: str) -> Any:
	"""Load a class from a dotted path like `package.module.ClassName`."""
	module_name, class_name = class_path.rsplit(".", 1)
	module = importlib.import_module(module_name)
	return getattr(module, class_name)


def _parse_json_dict(raw: str) -> Dict[str, Any]:
	"""Parse a JSON string into a dictionary."""
	if not raw:
		return {}
	parsed = json.loads(raw)
	if not isinstance(parsed, dict):
		raise ValueError("JSON args must decode to a dict")
	return parsed


def _inject_if_supported(target_cls: Any, kwargs: Dict[str, Any], optional_values: Dict[str, Any]) -> Dict[str, Any]:
	"""Inject optional constructor args only when supported by the target class."""
	sig = inspect.signature(target_cls.__init__)
	out = dict(kwargs)
	for key, value in optional_values.items():
		if key in sig.parameters and key not in out:
			out[key] = value
	return out


def _build_env(env_class: str, env_kwargs: Dict[str, Any], seed: int) -> BaseBanditEnv:
	"""Instantiate and validate an environment from user arguments."""
	env_cls = _load_class(env_class)
	kwargs = _inject_if_supported(env_cls, env_kwargs, {"seed": seed})
	env = env_cls(**kwargs)
	if not isinstance(env, BaseBanditEnv):
		raise TypeError(f"{env_class} is not a BaseBanditEnv")
	return env


def _build_behavior_policy(policy: str, policy_kwargs: Dict[str, Any], env: BaseBanditEnv, seed: int) -> BaseBanditAlgorithm:
	"""Instantiate behavior policy (`random`, `linucb`, `lints`, or custom class path)."""
	n_actions = int(env.n_actions)
	context_dim = int(env.context_dim)

	if policy.lower() == "random":
		kwargs = dict(policy_kwargs)
		kwargs.setdefault("seed", seed)
		return RandomPolicy(n_actions=n_actions, **kwargs)

	if policy.lower() == "linucb":
		kwargs = dict(policy_kwargs)
		return LinUCB(n_actions=n_actions, context_dim=context_dim, **kwargs)

	if policy.lower() == "lints":
		kwargs = dict(policy_kwargs)
		kwargs.setdefault("seed", seed)
		return LinTS(n_actions=n_actions, context_dim=context_dim, **kwargs)

	policy_cls = _load_class(policy)
	kwargs = _inject_if_supported(
		policy_cls,
		policy_kwargs,
		{"n_actions": n_actions, "context_dim": context_dim, "seed": seed},
	)
	algo = policy_cls(**kwargs)
	if not isinstance(algo, BaseBanditAlgorithm):
		raise TypeError(f"{policy} is not a BaseBanditAlgorithm")
	return algo


def rollout_dataset(env: BaseBanditEnv, behavior_policy: BaseBanditAlgorithm, n_steps: int) -> OfflineBanditDataset:
	"""Collect logged interactions and return a unified offline dataset."""
	contexts, actions, rewards = [], [], []
	optimal_rewards, expected_rewards = [], []
	has_optimal = True
	has_expected = True

	context = env.reset()
	for _ in range(n_steps):
		action = behavior_policy.select_action(context)
		next_context, reward, done, info = env.step(action)

		contexts.append(np.asarray(context, dtype=np.float32))
		actions.append(int(action))
		rewards.append(float(reward))

		if "optimal_reward" in info:
			optimal_rewards.append(float(info["optimal_reward"]))
		else:
			has_optimal = False

		if "expected_reward" in info:
			expected_rewards.append(float(info["expected_reward"]))
		else:
			has_expected = False

		behavior_policy.update(context, action, reward, info)

		if done or next_context is None:
			context = env.reset()
		else:
			context = next_context

	metadata = {
		"n_steps": n_steps,
		"n_actions": int(env.n_actions),
		"context_dim": int(env.context_dim),
		"behavior_policy": behavior_policy.__class__.__name__,
		"env": env.__class__.__name__,
	}

	return OfflineBanditDataset(
		contexts=np.asarray(contexts, dtype=np.float32),
		actions=np.asarray(actions, dtype=np.int64),
		rewards=np.asarray(rewards, dtype=np.float32),
		optimal_rewards=np.asarray(optimal_rewards, dtype=np.float32) if has_optimal else None,
		expected_rewards=np.asarray(expected_rewards, dtype=np.float32) if has_expected else None,
		metadata=metadata,
	)


def parse_args() -> argparse.Namespace:
	"""Define command-line arguments for offline data generation."""
	parser = argparse.ArgumentParser(description="Generate offline dataset from a bandit environment.")
	parser.add_argument("--env_class", type=str, default="envs.synthetic_env.LinearSyntheticEnv")
	parser.add_argument("--env_kwargs", type=str, default="{}", help="JSON dict for environment constructor args")
	parser.add_argument("--behavior_policy", type=str, default="random", help="random | linucb | lints | full.class.path")
	parser.add_argument("--policy_kwargs", type=str, default="{}", help="JSON dict for behavior policy constructor args")
	parser.add_argument("--n_steps", type=int, default=50000)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output_path", type=str, default="logs/offline_dataset.npz")
	parser.add_argument("--test_size", type=float, default=0.0, help="Optional train/test split ratio in (0,1)")
	parser.add_argument("--train_output_path", type=str, default="")
	parser.add_argument("--test_output_path", type=str, default="")
	parser.add_argument("--simulator_output_path", type=str, default="", help="Optional pickle path for RejectionSamplingEnv")
	parser.add_argument("--simulator_from", type=str, default="train", choices=["full", "train"], help="Which dataset to serialize as simulator")
	return parser.parse_args()


def main() -> None:
	"""Entrypoint: build env/policy, generate data, and persist outputs."""
	args = parse_args()

	env_kwargs = _parse_json_dict(args.env_kwargs)
	policy_kwargs = _parse_json_dict(args.policy_kwargs)

	env = _build_env(args.env_class, env_kwargs, seed=args.seed)
	behavior_policy = _build_behavior_policy(args.behavior_policy, policy_kwargs, env, seed=args.seed)

	dataset = rollout_dataset(env=env, behavior_policy=behavior_policy, n_steps=args.n_steps)
	dataset.save_npz(args.output_path)

	train_dataset = dataset
	if args.test_size > 0:
		train_dataset, test_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)

		output_path = Path(args.output_path)
		default_train = output_path.with_name(f"{output_path.stem}_train.npz")
		default_test = output_path.with_name(f"{output_path.stem}_test.npz")

		train_path = args.train_output_path or str(default_train)
		test_path = args.test_output_path or str(default_test)
		train_dataset.save_npz(train_path)
		test_dataset.save_npz(test_path)

	if args.simulator_output_path:
		source_dataset = dataset if args.simulator_from == "full" else train_dataset
		build_simulator(source_dataset, output_path=args.simulator_output_path)

	print(f"Saved dataset: {args.output_path}")
	if args.test_size > 0:
		print(f"Train/Test split complete. train={train_dataset.n_samples}, test={dataset.n_samples - train_dataset.n_samples}")
	if args.simulator_output_path:
		print(f"Saved simulator: {args.simulator_output_path}")


if __name__ == "__main__":
	main()
