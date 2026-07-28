"""

"""
import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from envs.synthetic_env import LinearSyntheticEnv
from algorithms.linucb import LinUCB
from evaluation.runner import EvaluatorRunner

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Bandit Benchmark Evaluator - MVP Test")
    parser.add_argument("--n_steps", type=int, default=2000, 
                        help="Interaction steps (Time horizons)")
    parser.add_argument("--n_actions", type=int, default=5, 
                        help="Number of available actions")
    parser.add_argument("--context_dim", type=int, default=10, 
                        help="Dimensionality of context features")
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="Exploration parameter for LinUCB")
    parser.add_argument("--log_path", type=str, default="logs/online_synthetic_run.npz",
                        help="Path to save runner logs (.npz)")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.info("===========================================")
    logging.info("Start Bandit Benchmark - Synthetic Environment MVP Test")
    logging.info(f"Parameter configuration: n_steps={args.n_steps}, n_actions={args.n_actions}, context_dim={args.context_dim}")
    logging.info("===========================================\n")
    
    # 1. Environment Initialization
    logging.info("Initializing LinearSyntheticEnv...")
    env = LinearSyntheticEnv(n_actions=args.n_actions, context_dim=args.context_dim, seed=42)
    
    # 2. Algorithm Initialization (LinUCB)
    logging.info("Initializing LinUCB algorithm...")
    algo = LinUCB(n_actions=args.n_actions, context_dim=args.context_dim, alpha=args.alpha)
    
    # 3. Instantiate EvaluatorRunner and run experiment
    logging.info(f"Starting experiment with total steps: {args.n_steps}...")
    runner = EvaluatorRunner(env=env, algo=algo, n_steps=args.n_steps)
    results = runner.run_experiment(log_path=args.log_path)
    logging.info(f"Saved run logs to: {args.log_path}")
    
    # 4. Print final results evaluation
    cumulative_regret = results.get("cumulative_regret")
    if cumulative_regret is not None:
        final_regret = cumulative_regret[-1]
        logging.info(f"Experiment completed!")
        logging.info(f"Final Cumulative Regret: {final_regret:.2f}")
        logging.info(f"Average Regret per Step: {final_regret/args.n_steps:.4f}")
        
        logging.info(f"Regret @ step 500 : {cumulative_regret[499]:.2f}")
        logging.info(f"Regret @ step 1000: {cumulative_regret[999]:.2f}")
        logging.info(f"Regret @ step 2000: {cumulative_regret[1999]:.2f}")
    else:
        logging.warning("Unable to calculate Regret, please check if the environment's info dictionary returned optimal_reward.")

if __name__ == "__main__":
    main()