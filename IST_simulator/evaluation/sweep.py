from __future__ import annotations
import os
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from datasets.ist_loader import default_config, load_ist_offline_logs
from envs.rejection_sampling_env import RejectionSamplingEnv
from evaluation.metrics import ensure_dir, summarize_df, make_basic_plots

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


def run_on_site(
    site_log: dict,
    algo: Any,
    *,
    seed: int = 0,
    shuffle: bool = False,
    stop_mode: str = "accepted_steps",
    n_accepted_steps: int = 200,
    max_total_steps: int = 1_000_000,
) -> Dict[str, float]:
    env = RejectionSamplingEnv(site_log, shuffle=shuffle, seed=seed, auto_reset=False)
    x = env.reset()

    rewards = []
    accepted = 0
    total = 0
    done = False

    while not done and total < max_total_steps:
        x_prev = x
        a = algo.select_action(x_prev)
        x, r, done, info = env.step(a)
        total += 1

        if info["accepted"]:
            algo.update(x_prev, a, r)
            rewards.append(float(r))
            accepted += 1

            if stop_mode == "accepted_steps" and accepted >= n_accepted_steps:
                break

    mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else float("nan")
    acceptance_rate = accepted / total if total > 0 else 0.0

    return {
        "mean_reward": mean_reward,
        "acceptance_rate": float(acceptance_rate),
        "n_accepted": int(accepted),
        "n_total": int(total),
    }


#run on pooled train
def run_on_pooled_train(
    train_log: dict,
    algo: Any,
    *,
    seed: int = 0,
    shuffle: bool = False,
    stop_mode: str = "accepted_steps",
    n_accepted_steps: int = 5000,  
) -> Dict[str, float]:
    env = RejectionSamplingEnv(train_log, shuffle=shuffle, seed=seed, auto_reset=False)
    x = env.reset()

    rewards = []
    accepted = 0
    total = 0
    done = False

    while not done:
        x_prev = x
        a = algo.select_action(x_prev)
        x, r, done, info = env.step(a)
        total += 1

        if info["accepted"]:
            algo.update(x_prev, a, r)
            rewards.append(float(r))
            accepted += 1
            if stop_mode == "accepted_steps" and accepted >= n_accepted_steps:
                break

    mean_reward = float(np.mean(rewards)) if rewards else float("nan")
    acceptance_rate = accepted / total if total else 0.0
    return {
        "mean_reward": mean_reward,
        "acceptance_rate": float(acceptance_rate),
        "n_accepted": int(accepted),
        "n_total": int(total),
    }


def algo_grid(K: int, d: int, base_seed: int) -> List[Tuple[str, Dict[str, Any], Any]]:
    grid = []

    grid.append((
        "RandomPolicy",
        {},
        lambda site_id: RandomPolicy(n_actions=K, seed=base_seed + site_id),
    ))

    for alpha in [0.1, 0.5, 1.0, 2.0]:
        hp = {"alpha": alpha}
        grid.append((
            "LinUCB",
            hp,
            lambda site_id, a=alpha: LinUCB(n_actions=K, context_dim=d, alpha=a),
        ))

    for lam in [0.1, 1.0, 10.0]:
        for sigma in [0.1, 0.5, 1.0]:
            hp = {"lam": lam, "sigma": sigma}
            grid.append((
                "LinearTS",
                hp,
                lambda site_id, l=lam, s=sigma: LinearTS(
                    n_actions=K, context_dim=d, lam=l, sigma=s, seed=base_seed + site_id
                ),
            ))

    return grid


def feature_settings() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("baseline", {"include_hospital_hash_feature": False}),
        ("baseline_plus_sitehash32", {"include_hospital_hash_feature": True, "hospital_hash_dim": 32}),
    ]


#traning modes: per site & pooled
def train_modes() -> List[str]:
    return ["per_site", "pooled"]


def run_sweep(
    ist_path: str = "data/IST_corrected.txt",
    out_dir: str = "results",
    seed: int = 0,
    split_seeds: List[int] | None = None,
    stop_mode: str = "accepted_steps",
    n_accepted_steps: int = 200,
    shuffle: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs full sweep and writes CSVs + plots

    multi split-seed support via split_seeds
    """
    ensure_dir(out_dir)
    if split_seeds is None:
        split_seeds = [0, 1, 2, 3, 4]

    all_rows = []

    for feat_name, feat_overrides in feature_settings():
        cfg0 = default_config()

        for split_seed in split_seeds:
            #seed split
            cfg = replace(cfg0, split_seed=split_seed, **feat_overrides)

            train_log, test_log, train_sites, test_sites = load_ist_offline_logs(ist_path, cfg)
            K = len(train_log["action_names"])
            d = train_log["contexts"].shape[1]

            algos = algo_grid(K, d, seed)

            for algo_name, hp, make_algo in algos:
                hp_str = ",".join([f"{k}={v}" for k, v in hp.items()]) if hp else ""

                for train_mode in train_modes():

                    # TRAIN
                    if train_mode == "per_site":
                        for site_id in train_sites:
                            site_id = int(site_id)
                            site_log = filter_log_to_site(train_log, site_id)
                            if len(site_log["actions"]) == 0:
                                continue

                            algo = make_algo(site_id)
                            res = run_on_site(
                                site_log,
                                algo,
                                seed=seed + site_id,
                                shuffle=shuffle,
                                stop_mode=stop_mode,
                                n_accepted_steps=n_accepted_steps,
                            )

                            all_rows.append({
                                "split_seed": split_seed,
                                "feature_setting": feat_name,
                                "train_mode": train_mode,
                                "phase": "train",
                                "site_id": site_id,
                                "algo": algo_name,
                                "hyperparams": hp_str,
                                **res,
                            })

                    elif train_mode == "pooled":
                        # one simulator for TRAIN: pool all train sites together
                        algo = make_algo(0)  # deterministic
                        res = run_on_pooled_train(
                            train_log,
                            algo,
                            seed=seed + 999,
                            shuffle=shuffle,
                            stop_mode=stop_mode,
                            n_accepted_steps=5000,  # pooled needs a bigger budget
                        )

                        all_rows.append({
                            "split_seed": split_seed,
                            "feature_setting": feat_name,
                            "train_mode": train_mode,
                            "phase": "train",
                            "site_id": -1,              # -1 means pooled
                            "algo": algo_name,
                            "hyperparams": hp_str,
                            **res,
                        })

                    else:
                        raise ValueError(f"Unknown train_mode: {train_mode}")

                    # TEST 
                    for site_id in test_sites:
                        site_id = int(site_id)
                        site_log = filter_log_to_site(test_log, site_id)
                        if len(site_log["actions"]) == 0:
                            continue

                        algo = make_algo(site_id)
                        res = run_on_site(
                            site_log,
                            algo,
                            seed=seed + 100000 + site_id,
                            shuffle=shuffle,
                            stop_mode=stop_mode,
                            n_accepted_steps=n_accepted_steps,
                        )

                        all_rows.append({
                            "split_seed": split_seed,
                            "feature_setting": feat_name,
                            "train_mode": train_mode,
                            "phase": "test",
                            "site_id": site_id,
                            "algo": algo_name,
                            "hyperparams": hp_str,
                            **res,
                        })

            print(
                f"[done] feature_setting={feat_name} split_seed={split_seed} | "
                f"K={K} d={d} | train_sites={len(train_sites)} test_sites={len(test_sites)}"
            )

    per_site_df = pd.DataFrame(all_rows)
    per_site_path = os.path.join(out_dir, "ist_per_site_results.csv")
    per_site_df.to_csv(per_site_path, index=False)
    print(f"Wrote per-site results to: {per_site_path}")

    summary_df = summarize_df(
        per_site_df,
        value_col="mean_reward",
        group_cols=["split_seed", "feature_setting", "train_mode", "algo", "hyperparams", "phase"],
    )
    summary_path = os.path.join(out_dir, "ist_summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary results to: {summary_path}")

    plot_dir = os.path.join(out_dir, "plots")
    for (feat_name, split_seed, train_mode), g in per_site_df.groupby(["feature_setting", "split_seed", "train_mode"]):
        make_basic_plots(
            g,
            out_dir=os.path.join(plot_dir, f"{feat_name}_seed{split_seed}_{train_mode}"),
            value_col="mean_reward",
        )
    print(f"Wrote plots to: {plot_dir}")

    return per_site_df, summary_df


if __name__ == "__main__":
    run_sweep(
        ist_path="data/IST_corrected.txt",
        out_dir="results",
        seed=0,
        split_seeds=[0, 1, 2, 3, 4],
        stop_mode="accepted_steps",
        n_accepted_steps=200,
        shuffle=False,
    )