import numpy as np
import logging
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from envs.synthetic_env import LinearSyntheticEnv
from envs.data_driven_env import RejectionSamplingEnv
from algorithms.linucb import LinUCB
from algorithms.base_algo import BaseBanditAlgorithm
from data.dataset import OfflineBanditDataset
from evaluation.runner import EvaluatorRunner
from evaluation.visualization import plot_cumulative_reward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- random policy for data collection ---
class RandomPolicy(BaseBanditAlgorithm):
    def select_action(self, context: np.ndarray) -> int:
        return np.random.randint(self.n_actions)
    def update(self, context, action, reward, info=None):
        pass # random policy does not learn

def generate_offline_data(n_samples: int, n_actions: int, context_dim: int) -> dict:
    """Collect historical logs by running a random policy in the real environment"""
    logging.info(f"Collecting {n_samples} offline log data using Random Policy...")
    env = LinearSyntheticEnv(n_actions=n_actions, context_dim=context_dim, seed=123)
    algo = RandomPolicy(n_actions=n_actions)
    
    contexts, actions, rewards, optimal_rewards, expected_rewards = [], [], [], [], []
    context = env.reset()
    
    for _ in range(n_samples):
        action = algo.select_action(context)
        next_context, reward, done, info = env.step(action)
        
        contexts.append(context)
        actions.append(action)
        rewards.append(reward)
        optimal_rewards.append(info['optimal_reward'])
        expected_rewards.append(info['expected_reward'])
        
        context = next_context
        
    return {
        'contexts': np.array(contexts),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'optimal_rewards': np.array(optimal_rewards),
        'expected_rewards': np.array(expected_rewards)
    }


def load_dataset_npz_compat(path: str) -> OfflineBanditDataset:
    """Load dataset and tolerate legacy metadata pickle incompatibilities."""
    with np.load(path, allow_pickle=True) as data:
        payload = {
            "contexts": data["contexts"],
            "actions": data["actions"],
            "rewards": data["rewards"],
        }
        for optional_key in ("optimal_rewards", "expected_rewards", "metadata"):
            if optional_key not in data.files:
                continue
            try:
                payload[optional_key] = data[optional_key]
            except ModuleNotFoundError:
                logging.warning(
                    "Skipping legacy key '%s' due to pickle incompatibility in %s",
                    optional_key,
                    path,
                )
    return OfflineBanditDataset.from_dict(payload)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline rejection-sampling evaluation pipeline from dataset .npz or generated data."
    )
    parser.add_argument("--dataset_path", type=str, default="logs/offline_dataset.npz")
    parser.add_argument("--log_path", type=str, default="logs/offline_rejection_run.npz")
    parser.add_argument("--plot_path", type=str, default="logs/offline_rejection_cum_reward.png")
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--generate_if_missing", action="store_true")
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--n_actions", type=int, default=5)
    parser.add_argument("--context_dim", type=int, default=10)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)

    # 1. Load historical data from .npz (preferred), fallback to in-script generation if requested.
    if dataset_path.exists():
        logging.info(f"Loading offline dataset from: {dataset_path}")
        offline_data = load_dataset_npz_compat(str(dataset_path))
    elif args.generate_if_missing:
        logging.info(
            f"Dataset not found at {dataset_path}. Generating fallback data "
            f"(n_samples={args.n_samples}, n_actions={args.n_actions}, context_dim={args.context_dim})."
        )
        offline_data = generate_offline_data(args.n_samples, args.n_actions, args.context_dim)
    else:
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            "Run data/gen_data.py first or use --generate_if_missing."
        )

    if isinstance(offline_data, dict):
        n_actions = int(len(np.unique(offline_data["actions"])))
        context_dim = int(offline_data["contexts"].shape[1])
    else:
        n_actions = int(offline_data.n_actions)
        context_dim = int(offline_data.context_dim)

    # 2. Wrap historical data into offline environment
    logging.info("Initializing offline evaluation environment (RejectionSamplingEnv)...")
    eval_env = RejectionSamplingEnv(offline_data)

    # 3. Initialize target algorithm
    logging.info("Initializing target algorithm LinUCB...")
    target_algo = LinUCB(n_actions=n_actions, context_dim=context_dim, alpha=args.alpha)

    # 4. Execute evaluation
    logging.info(f"Starting offline evaluation! Looking for {args.eval_steps} valid interactions...")
    runner = EvaluatorRunner(env=eval_env, algo=target_algo, n_steps=args.eval_steps)
    results = runner.run_experiment(log_path=args.log_path)
    logging.info(f"Saved run logs to: {args.log_path}")

    if not args.no_plot:
        plot_cumulative_reward(
            log_path=args.log_path,
            output_path=args.plot_path,
            show=not args.no_show,
        )
        logging.info(f"Saved visualization to: {args.plot_path}")

    # 5. Analyze results
    valid_steps = results['valid_steps'][0]
    total_interactions = results['total_interactions'][0]
    
    logging.info("\n================ Evaluation Report ================")
    logging.info(f"Target valid steps: {args.eval_steps}")
    logging.info(f"Actual collected valid steps: {valid_steps}")
    logging.info(f"Total historical log lines consumed: {total_interactions}")
    logging.info(f"Data utilization rate (Acceptance Rate): {valid_steps / total_interactions:.2%}")
    
    if 'cumulative_regret' in results and len(results['cumulative_regret']) > 0:
        final_regret = results['cumulative_regret'][-1]
        logging.info(f"Final cumulative regret of pure offline learning: {final_regret:.2f}")

if __name__ == "__main__":
    main()
