import numpy as np
import logging
from typing import Any, Callable, Dict, List, Tuple
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs" / "meta_syn_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "RUNNING_LOGS.txt"

from envs.synthetic_env import LinearSyntheticEnv
from envs.data_driven_env import build_simulator
from evaluation.runner import EvaluatorRunner
from algorithms.base_algo import BaseBanditAlgorithm
from algorithms.linucb import LinUCB
from algorithms.lints import LinTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
    ],
    force=True,
)


# --- RandomPolicy: baseline, not learning, just for data collection ---
class RandomPolicy(BaseBanditAlgorithm):
    def select_action(self, context: np.ndarray) -> int:
        return np.random.randint(self.n_actions)

    def update(self, context, action, reward, info=None):
        pass


# ==========================================
# Core Component 1: Sampling Site-specific Thetas (P_theta) for Multi-site Similarity & Diversity
# ==========================================
def sample_site_thetas(n_actions: int, context_dim: int, site_seed: int) -> np.ndarray:
    """
    simulate sampling from a distribution P_theta that generates site-specific reward parameters (thetas).
    """
    rng = np.random.default_rng(site_seed)
    # assume all sites share a common base structure (Base Thetas) + site-specific perturbation
    base_rng = np.random.default_rng(42)
    base_thetas = base_rng.uniform(-0.5, 0.5, size=(n_actions, context_dim))

    # add Gaussian noise to create site-specific variations, controlling the variance to ensure some similarity but also diversity
    site_perturbation = rng.normal(0, 0.2, size=(n_actions, context_dim))

    raw_thetas = base_thetas + site_perturbation
    # L2 normalization to control the reward range
    return raw_thetas / np.linalg.norm(raw_thetas, axis=1, keepdims=True)


# ==========================================
# Core Component 2: Data Collection and Merging
# ==========================================
def collect_and_merge_data(n_sites: int, samples_per_site: int, n_actions: int, context_dim: int) -> Dict[str, np.ndarray]:
    """Collect data from multiple sites and merge it into a super offline dataset"""
    logging.info(f"Collecting data from {n_sites} different Training Sites...")
    all_contexts, all_actions, all_rewards = [], [], []
    all_opt_rewards, all_exp_rewards = [], []

    for site_idx in range(n_sites):
        # 1. Sample the theta_i for the site
        site_thetas = sample_site_thetas(n_actions, context_dim, site_seed=100 + site_idx)
        # 2. Instantiate the true environment for the site
        env = LinearSyntheticEnv(n_actions, context_dim, true_thetas=site_thetas, seed=100 + site_idx)
        algo = RandomPolicy(n_actions)

        context = env.reset()
        for _ in range(samples_per_site):
            action = algo.select_action(context)
            next_context, reward, done, info = env.step(action)

            all_contexts.append(context)
            all_actions.append(action)
            all_rewards.append(reward)
            all_opt_rewards.append(info['optimal_reward'])
            all_exp_rewards.append(info['expected_reward'])

            context = next_context

    # Merge into a large dictionary
    return {
        'contexts': np.array(all_contexts),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'optimal_rewards': np.array(all_opt_rewards),
        'expected_rewards': np.array(all_exp_rewards),
    }


def tune_algorithm(
    algo_name: str,
    param_name: str,
    candidates: List[float],
    algo_factory: Callable[[float], BaseBanditAlgorithm],
    offline_simulator,
    train_steps: int,
) -> Tuple[float, float]:
    """Tune one hyperparameter for an algorithm in the offline simulator."""
    best_param = None
    best_offline_regret = float('inf')

    for param in candidates:
        logging.info(f"Evaluating {algo_name} ({param_name}={param})...")
        algo = algo_factory(param)

        offline_simulator.reset()
        runner = EvaluatorRunner(env=offline_simulator, algo=algo, n_steps=train_steps)
        results = runner.run_experiment()

        if len(results.get('cumulative_regret', [])) > 0:
            final_regret = results['cumulative_regret'][-1]
            logging.info(
                f"  -> Valid Steps: {results['valid_steps'][0]}, Final Offline Regret: {final_regret:.2f}"
            )
            if final_regret < best_offline_regret:
                best_offline_regret = final_regret
                best_param = param

    logging.info(
        f"--> {algo_name} Offline Hyperparameter Search Completed. "
        f"Best {param_name}: {best_param}, with Offline Regret: {best_offline_regret:.2f}"
    )
    return best_param, best_offline_regret


def evaluate_algorithm_on_sites(
    algo_name: str,
    best_param: float,
    algo_factory: Callable[[float], BaseBanditAlgorithm],
    n_eval_sites: int,
    n_actions: int,
    context_dim: int,
    eval_steps: int,
    log_suffix: str,
) -> List[float]:
    """Evaluate a tuned algorithm on unseen true sites."""
    eval_regrets = []

    for eval_idx in range(n_eval_sites):
        eval_seed = 500 + eval_idx
        new_site_thetas = sample_site_thetas(n_actions, context_dim, site_seed=eval_seed)
        true_env = LinearSyntheticEnv(n_actions, context_dim, true_thetas=new_site_thetas, seed=eval_seed)

        algo = algo_factory(best_param)
        log_file = f"logs/meta_syn_logs/eval_site_{eval_idx}_{log_suffix}.npz"
        runner = EvaluatorRunner(env=true_env, algo=algo, n_steps=eval_steps)
        results = runner.run_experiment(log_path=log_file)

        site_final_regret = results['cumulative_regret'][-1]
        eval_regrets.append(site_final_regret)
        logging.info(f"{algo_name} Site {eval_idx} evaluation completed. Final Regret: {site_final_regret:.2f}")

    return eval_regrets


# ==========================================
# Main Pipeline (Meta-Pipeline)
# ==========================================
def main():
    # --- Hyperparameter Settings ---
    N_ACTIONS = 5
    CONTEXT_DIM = 10

    N_TRAIN_SITES = 10  # Number of training sites during training phase
    SAMPLES_PER_SITE = 5000  # Number of logs collected from each training site (Total 50000 logs)

    N_EVAL_SITES = 20  # Number of evaluation sites during testing phase
    EVAL_STEPS = 1000  # Number of online interaction steps for each testing site
    OFFLINE_TRAIN_STEPS = 2000

    logging.info("=========== Stage 1: Offline Multi-site Data Collection and Simulator Construction ===========")
    merged_data = collect_and_merge_data(N_TRAIN_SITES, SAMPLES_PER_SITE, N_ACTIONS, CONTEXT_DIM)

    # Use the new data_driven_env to build and save the merged simulator
    simulator_path = "data/synthetic_configs/merged_simulator.pkl"
    logging.info(f"Building super offline simulator and saving to {simulator_path} ...")
    offline_simulator = build_simulator(merged_data, output_path=simulator_path)

    algorithm_configs: List[Dict[str, Any]] = [
        {
            "name": "LinUCB",
            "param_name": "alpha",
            "candidates": [0.1, 0.5, 1.0, 2.0],
            "log_suffix": "linucb",
            "factory": lambda p: LinUCB(n_actions=N_ACTIONS, context_dim=CONTEXT_DIM, alpha=p),
        },
        {
            "name": "LinTS",
            "param_name": "sigma",
            "candidates": [0.1, 0.5, 1.0, 2.0],
            "log_suffix": "lints",
            "factory": lambda p: LinTS(n_actions=N_ACTIONS, context_dim=CONTEXT_DIM, lam=1.0, sigma=p, seed=42),
        },
    ]

    summary_stats: Dict[str, Dict[str, float]] = {}

    for cfg in algorithm_configs:
        logging.info(
            f"\n=========== Stage 2: Hyperparameter Search for {cfg['name']} in Offline Simulator (Training) ==========="
        )
        best_param, _ = tune_algorithm(
            algo_name=cfg["name"],
            param_name=cfg["param_name"],
            candidates=cfg["candidates"],
            algo_factory=cfg["factory"],
            offline_simulator=offline_simulator,
            train_steps=OFFLINE_TRAIN_STEPS,
        )

        logging.info(
            f"\n=========== Stage 3: {cfg['name']} Generalization Evaluation on Novel True Site Distributions ==========="
        )
        eval_regrets = evaluate_algorithm_on_sites(
            algo_name=cfg["name"],
            best_param=best_param,
            algo_factory=cfg["factory"],
            n_eval_sites=N_EVAL_SITES,
            n_actions=N_ACTIONS,
            context_dim=CONTEXT_DIM,
            eval_steps=EVAL_STEPS,
            log_suffix=cfg["log_suffix"],
        )

        mean_regret = float(np.mean(eval_regrets))
        std_regret = float(np.std(eval_regrets))
        summary_stats[cfg["name"]] = {
            "mean": mean_regret,
            "std": std_regret,
            "best_param": best_param,
        }

        logging.info("\n=========== Final Report across Sites ===========")
        logging.info(f"{cfg['name']} mean regret across {N_EVAL_SITES} sites: {mean_regret:.2f}")
        logging.info(f"{cfg['name']} std regret across sites: {std_regret:.2f}")

    linucb_mean = summary_stats["LinUCB"]["mean"]
    lints_mean = summary_stats["LinTS"]["mean"]

    logging.info("\n=========== LinUCB vs LinTS Comparison ===========")
    logging.info(f"LinUCB mean/std regret: {summary_stats['LinUCB']['mean']:.2f} / {summary_stats['LinUCB']['std']:.2f}")
    logging.info(f"LinTS  mean/std regret: {summary_stats['LinTS']['mean']:.2f} / {summary_stats['LinTS']['std']:.2f}")
    if lints_mean < linucb_mean:
        logging.info(f"LinTS wins by {linucb_mean - lints_mean:.2f} mean regret.")
    elif lints_mean > linucb_mean:
        logging.info(f"LinUCB wins by {lints_mean - linucb_mean:.2f} mean regret.")
    else:
        logging.info("LinUCB and LinTS tie on mean regret.")


if __name__ == "__main__":
    main()
