from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SummaryStats:
    n_sites_total: int
    n_sites_finite: int
    mean: float
    median: float
    std: float
    p10: float
    p25: float
    p75: float
    p90: float
    min: float
    max: float

    def to_dict(self) -> Dict:
        return {
            "n_sites_total": self.n_sites_total,
            "n_sites_finite": self.n_sites_finite,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "p10": self.p10,
            "p25": self.p25,
            "p75": self.p75,
            "p90": self.p90,
            "min": self.min,
            "max": self.max,
        }


def summarize_array(values: Iterable[float]) -> SummaryStats:
    vals = list(values)
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        # Return NaNs but keep counts
        return SummaryStats(
            n_sites_total=len(arr),
            n_sites_finite=0,
            mean=float("nan"),
            median=float("nan"),
            std=float("nan"),
            p10=float("nan"),
            p25=float("nan"),
            p75=float("nan"),
            p90=float("nan"),
            min=float("nan"),
            max=float("nan"),
        )

    return SummaryStats(
        n_sites_total=len(arr),
        n_sites_finite=int(finite.size),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        std=float(np.std(finite)),
        p10=float(np.quantile(finite, 0.10)),
        p25=float(np.quantile(finite, 0.25)),
        p75=float(np.quantile(finite, 0.75)),
        p90=float(np.quantile(finite, 0.90)),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
    )


def summarize_df(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Summarize df[value_col] for each group defined by group_cols.
    Returns a new DataFrame with summary stats per group.
    """
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stats = summarize_array(g[value_col].tolist()).to_dict()
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Plotting 


def plot_histogram(values: Iterable[float], title: str, out_path: str, bins: int = 20) -> None:
    """
    histogram
    """
    import matplotlib.pyplot as plt

    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return

    plt.figure()
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_box(values_by_label: Dict[str, List[float]], title: str, out_path: str) -> None:
    """
   boxplot across labels
    """
    import matplotlib.pyplot as plt

    labels = list(values_by_label.keys())
    data = [np.asarray([v for v in values_by_label[l] if np.isfinite(v)], dtype=float) for l in labels]
    if all(d.size == 0 for d in data):
        return

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_basic_plots(
    per_site_df: pd.DataFrame,
    out_dir: str,
    value_col: str = "mean_reward",
) -> None:
    """
    Creates:
      - histogram per (algo, phase)
      - boxplot comparing algos within each phase
    Expects per_site_df columns: ['algo', 'phase', value_col]
    """
    ensure_dir(out_dir)

    # Histograms per (algo, phase)
    for (algo, phase), g in per_site_df.groupby(["algo", "phase"]):
        out_path = os.path.join(out_dir, f"hist_{value_col}_{algo}_{phase}.png".replace(" ", "_"))
        plot_histogram(g[value_col].tolist(), f"{value_col} | {algo} | {phase}", out_path)

    # Boxplots comparing algos, per phase
    for phase, g in per_site_df.groupby(["phase"]):
        values_by_algo = {algo: gg[value_col].tolist() for algo, gg in g.groupby("algo")}
        out_path = os.path.join(out_dir, f"box_{value_col}_{phase}.png".replace(" ", "_"))
        plot_box(values_by_algo, f"{value_col} by algo | {phase}", out_path)