
"""
- Current action space is the 2x3 factorial = 6 arms: (use 4 arms instead??)
    RXASP ∈ {N,Y} and RXHEP ∈ {N,L,M} where H is merged into M (pilot coding).
- Supports alternative action spaces:
    - "heparin4" (N/L/M/H) = 4 arms
    - "aspirin2" = 2 arms
- Supports multiple reward definitions, including:
    - survival_6m
    - survival_14d
    - occode_utility 
    - death_or_dependence_6m
    - safety_composite_14d (survival minus event penalties)
- Returns BOTH:
    - a scalar `rewards` used by algorithms
    - a `reward_components` matrix with named components for analysis

Output offline_log format:
  {
    "contexts": np.ndarray (N, d) float32,
    "actions": np.ndarray (N,) int64,
    "rewards": np.ndarray (N,) float32,              
    "reward_components": np.ndarray (N, m) float32,   # extra components
    "reward_component_names": list[str],
    "sites": np.ndarray (N,) int64,                   # HOSPNUM
    "site_str": np.ndarray (N,) object,           
    "action_names": list[str],
    "feature_names": list[str],
    "meta": dict
  }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd



# Configuration
DEFAULT_BASELINE_COVARIATES: Tuple[str, ...] = (
    # Core demographics / timing / vitals
    "AGE",
    "SEX",
    "RDELAY",
    "RSBP",
    # Clinical presentation
    "RCONSC",
    "RSLEEP",
    "RATRIAL",
    "RCT",
    "RVISINF",
    "RHEP24",
    "RASP3",
    # Neurological deficits (Y/N/C)
    "RDEF1",
    "RDEF2",
    "RDEF3",
    "RDEF4",
    "RDEF5",
    "RDEF6",
    "RDEF7",
    "RDEF8",
    # Stroke subtype (categorical)
    "STYPE",
)

DEFAULT_REWARD_COMPONENTS: Tuple[str, ...] = (
    "survival_14d",
    "survival_6m",
    "occode_utility",
    "death_or_dependence_6m",
    "H14",      # cerebral bleed within 14d
    "TRAN14",   # major non-cerebral bleed within 14d
    "MI14",
    "PE14",
    "ISC14",
)


@dataclass(frozen=True)
class ISTLoadConfig:
    # ---- Covariates 
    covariates: Tuple[str, ...] = DEFAULT_BASELINE_COVARIATES

    # If True, include COUNTRY (one-hot) in contexts in addition to baseline covariates
    include_country_feature: bool = False

    include_hospital_hash_feature: bool = False
    hospital_hash_dim: int = 32  # small, fixed

    # ---- Actions
    # "factorial6": RXASP ∈ {N,Y}, RXHEP ∈ {N,L,M} where H merged into M  -> 6 arms
    # "heparin4" : RXHEP ∈ {N,L,M,H}                                     -> 4 arms
    # "aspirin2" : RXASP ∈ {N,Y}                                         -> 2 arms
    action_mode: str = "factorial6"

    # ---- Rewards
    # Scalar reward used for learning:
    #   - "survival_6m"
    #   - "survival_14d"
    #   - "occode_utility"
    #   - "death_or_dependence_6m"
    #   - "safety_composite_14d"
    reward_mode: str = "death_or_dependence_6m"

    # Additional reward components to export for analysis (matrix)
    reward_components: Tuple[str, ...] = DEFAULT_REWARD_COMPONENTS

    # Safety composite weights (only used if reward_mode == "safety_composite_14d")
    w_h14: float = 1.0
    w_tran14: float = 1.0
    w_mi14: float = 0.5
    w_pe14: float = 0.5
    w_isc14: float = 0.2  # optional penalty for recurrent ischemic stroke indicator

    # ---- Site split
    site_col: str = "HOSPNUM"
    split_seed: int = 0
    train_site_frac: float = 0.5

    # ---- Filtering
    drop_missing_action: bool = True
    drop_missing_reward: bool = True
    drop_missing_covariates: bool = True

    # missing markers (IST file has lots of blanks and 99/999 style sentinels)
    na_values: Tuple[str, ...] = ("", " ", "NA", "NaN", "nan", "N/A", "U", "u", "99", "999", "9999")


def default_config() -> ISTLoadConfig:
    return ISTLoadConfig()



# Read IST table
def read_ist_table(path: str, cfg: Optional[ISTLoadConfig] = None) -> pd.DataFrame:
    """
    Reads IST_corrected.txt and returns a DataFrame with string columns.
    """
    cfg = cfg or default_config()
    df = pd.read_csv(
        path,
        sep="\t",
        engine="python",
        dtype=str,
        na_values=list(cfg.na_values),
        keep_default_na=True,
        encoding="utf-16le",  
    )
    df.columns = [c.strip() for c in df.columns]
    return df


# ----------------------------
# Encoding helpers
# ----------------------------

def _map_yn(series: pd.Series) -> pd.Series:
    """Map Y/N to 1/0; anything else -> NaN."""
    s = series.astype(str).str.upper()
    return s.map({"Y": 1.0, "N": 0.0})


def _encode_sex(series: pd.Series) -> pd.Series:
    """SEX: M/F -> 1/0; else NaN."""
    s = series.astype(str).str.upper()
    return s.map({"M": 1.0, "F": 0.0})


def _encode_rconsc(series: pd.Series) -> pd.DataFrame:
    """
    RCONSC: F/D/U (fully alert/drowsy/unconscious).
    One-hot with names: RCONSC_F, RCONSC_D, RCONSC_U
    """
    s = series.astype(str).str.upper()
    cats = ["F", "D", "U"]
    out = {}
    for c in cats:
        out[f"RCONSC_{c}"] = (s == c).astype(float)
    return pd.DataFrame(out)


def _encode_rdef(series: pd.Series, name: str) -> pd.DataFrame:
    """
    RDEFk: Y/N/C (can't assess).
    Encode as two indicators:
      - {name}_Y (deficit present)
      - {name}_C (can't assess)
    (N is baseline implied when both are 0)
    """
    s = series.astype(str).str.upper()
    return pd.DataFrame(
        {
            f"{name}_Y": (s == "Y").astype(float),
            f"{name}_C": (s == "C").astype(float),
        }
    )


def _one_hot(series: pd.Series, prefix: str, max_levels: Optional[int] = None) -> pd.DataFrame:
    """
    One-hot encode categorical string series (including NaNs as all-zeros).
    If max_levels is set, keep only the most frequent levels (others dropped).
    """
    s = series.astype("string")
    vc = s.value_counts(dropna=True)
    if max_levels is not None and len(vc) > max_levels:
        keep = set(vc.index[:max_levels].tolist())
        s = s.where(s.isin(list(keep)))
    d = pd.get_dummies(s, prefix=prefix, dummy_na=False)
    # Ensure float
    return d.astype(float)


def _hash_hospital_feature(hosp: pd.Series, dim: int) -> pd.DataFrame:
    """
    Hash HOSPNUM into a fixed-size one-hot bucket feature.
    """
    # Coerce to string tokens
    tokens = hosp.astype("string").fillna("NA")
    # Stable hash -> bucket
    buckets = tokens.apply(lambda x: (hash(str(x)) % dim)).astype(int)
    mat = np.zeros((len(buckets), dim), dtype=float)
    mat[np.arange(len(buckets)), buckets.to_numpy()] = 1.0
    cols = [f"HOSP_HASH_{i}" for i in range(dim)]
    return pd.DataFrame(mat, columns=cols)



# Context encoding

def encode_contexts(df: pd.DataFrame, cfg: ISTLoadConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Build context matrix X from cfg.covariates, plus optional COUNTRY and hospital hash.
    """
    parts: List[pd.DataFrame] = []
    feature_names: List[str] = []

    for col in cfg.covariates:
        if col not in df.columns:
            raise KeyError(f"Covariate '{col}' not found in IST data.")

        if col == "SEX":
            v = _encode_sex(df[col]).to_frame("SEX_M")
            parts.append(v)
            feature_names.extend(list(v.columns))
        elif col == "RCONSC":
            v = _encode_rconsc(df[col])
            parts.append(v)
            feature_names.extend(list(v.columns))
        elif col.startswith("RDEF"):
            v = _encode_rdef(df[col], col)
            parts.append(v)
            feature_names.extend(list(v.columns))
        elif col in ("RSLEEP", "RATRIAL", "RCT", "RVISINF", "RHEP24", "RASP3"):
            v = _map_yn(df[col]).to_frame(col)
            parts.append(v)
            feature_names.append(col)
        elif col == "STYPE":
            v = _one_hot(df[col].astype(str).str.upper(), prefix="STYPE")
            parts.append(v)
            feature_names.extend(list(v.columns))
        else:
            # numeric
            v = pd.to_numeric(df[col], errors="coerce").astype(float).to_frame(col)
            parts.append(v)
            feature_names.append(col)

    if cfg.include_country_feature:
        if "COUNTRY" not in df.columns:
            raise KeyError("COUNTRY not found in IST data.")
        v = _one_hot(df["COUNTRY"].astype(str).str.upper(), prefix="COUNTRY", max_levels=50)
        parts.append(v)
        feature_names.extend(list(v.columns))

    if cfg.include_hospital_hash_feature:
        if cfg.site_col not in df.columns:
            raise KeyError(f"{cfg.site_col} not found in IST data.")
        v = _hash_hospital_feature(df[cfg.site_col], cfg.hospital_hash_dim)
        parts.append(v)
        feature_names.extend(list(v.columns))

    X_df = pd.concat(parts, axis=1)
    X = X_df.to_numpy(dtype=np.float32)
    return X, feature_names


# Action encoding

def encode_actions(df: pd.DataFrame, cfg: ISTLoadConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Encode actions as integer-coded arms.

    - factorial6 (default):
        RXASP ∈ {N,Y} -> asp in {0,1}
        RXHEP ∈ {N,L,M,H} -> hep in {0,1,2} where H merged into M
        action_id = asp * 3 + hep  -> 0..5
    - heparin4:
        RXHEP ∈ {N,L,M,H} -> 0..3
    - aspirin2:
        RXASP ∈ {N,Y} -> 0..1
    """
    mode = cfg.action_mode.lower().strip()

    if mode == "factorial6":
        if "RXASP" not in df.columns or "RXHEP" not in df.columns:
            raise KeyError("Need RXASP and RXHEP for factorial6 actions.")
        asp = df["RXASP"].astype(str).str.upper().map({"N": 0, "Y": 1})
        hep_raw = df["RXHEP"].astype(str).str.upper()
        # Merge H -> M for pilot coding and collapse to 3 levels
        hep = hep_raw.replace({"H": "M"}).map({"N": 0, "L": 1, "M": 2})

        a = asp * 3 + hep
        action_names = [
            "ASP_N__HEP_N",
            "ASP_N__HEP_L",
            "ASP_N__HEP_M",
            "ASP_Y__HEP_N",
            "ASP_Y__HEP_L",
            "ASP_Y__HEP_M",
        ]
        return a.to_numpy(dtype="float64"), action_names

    if mode == "heparin4":
        if "RXHEP" not in df.columns:
            raise KeyError("Need RXHEP for heparin4 actions.")
        hep = df["RXHEP"].astype(str).str.upper().map({"N": 0, "L": 1, "M": 2, "H": 3})
        action_names = ["HEP_N", "HEP_L", "HEP_M", "HEP_H"]
        return hep.to_numpy(dtype="float64"), action_names

    if mode == "aspirin2":
        if "RXASP" not in df.columns:
            raise KeyError("Need RXASP for aspirin2 actions.")
        asp = df["RXASP"].astype(str).str.upper().map({"N": 0, "Y": 1})
        action_names = ["ASP_N", "ASP_Y"]
        return asp.to_numpy(dtype="float64"), action_names

    raise ValueError("action_mode must be one of: 'factorial6', 'heparin4', 'aspirin2'.")


# ----------------------------
# Reward encoding
# ----------------------------

def _reward_survival_6m(df: pd.DataFrame) -> pd.Series:
    if "FDEAD" not in df.columns:
        raise KeyError("FDEAD not found.")
    s = df["FDEAD"].astype(str).str.upper().map({"Y": 0.0, "N": 1.0})
    return s


def _reward_survival_14d(df: pd.DataFrame) -> pd.Series:
    if "ID14" not in df.columns:
        raise KeyError("ID14 not found.")
    id14 = pd.to_numeric(df["ID14"], errors="coerce").astype(float)
    return 1.0 - id14


def _reward_occode_utility(df: pd.DataFrame) -> pd.Series:
    """
    OCCODE: 1-dead / 2-dependent / 3-not recovered / 4-recovered / 8/9 missing
    Utility mapping (simple, monotone):
      1 -> 0.0
      2 -> 0.33
      3 -> 0.66
      4 -> 1.0
    """
    if "OCCODE" not in df.columns:
        raise KeyError("OCCODE not found.")
    oc = pd.to_numeric(df["OCCODE"], errors="coerce").astype(float)
    mapping = {1.0: 0.0, 2.0: 1.0 / 3.0, 3.0: 2.0 / 3.0, 4.0: 1.0}
    return oc.map(mapping)


def _reward_death_or_dependence_6m(df: pd.DataFrame) -> pd.Series:
    """
    Common primary endpoint: alive AND not dependent at 6 months
    Using OCCODE:
      success (reward=1): 3 or 4
      failure (reward=0): 1 or 2
    """
    if "OCCODE" not in df.columns:
        raise KeyError("OCCODE not found.")
    oc = pd.to_numeric(df["OCCODE"], errors="coerce").astype(float)
    r = pd.Series(np.nan, index=df.index, dtype=float)
    r.loc[oc.isin([3.0, 4.0])] = 1.0
    r.loc[oc.isin([1.0, 2.0])] = 0.0
    return r


def _reward_safety_composite_14d(df: pd.DataFrame, cfg: ISTLoadConfig) -> pd.Series:
    """
    Simple safety/efficacy composite at 14d:
      survival_14d - penalties for adverse events.
    """
    surv = _reward_survival_14d(df)

    def ind(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").astype(float)

    h14 = ind("H14")
    tran14 = ind("TRAN14")
    mi14 = ind("MI14")
    pe14 = ind("PE14")
    isc14 = ind("ISC14")

    # If any component is missing, result becomes NaN -> filtered if drop_missing_reward=True
    comp = surv - (
        cfg.w_h14 * h14
        + cfg.w_tran14 * tran14
        + cfg.w_mi14 * mi14
        + cfg.w_pe14 * pe14
        + cfg.w_isc14 * isc14
    )
    return comp


def encode_rewards(df: pd.DataFrame, cfg: ISTLoadConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      scalar_reward (float64 with NaNs before filtering),
      reward_components_matrix (float64 with NaNs),
      reward_component_names
    """
    # Scalar reward
    mode = cfg.reward_mode.lower().strip()
    if mode == "survival_6m":
        scalar = _reward_survival_6m(df)
    elif mode == "survival_14d":
        scalar = _reward_survival_14d(df)
    elif mode == "occode_utility":
        scalar = _reward_occode_utility(df)
    elif mode == "death_or_dependence_6m":
        scalar = _reward_death_or_dependence_6m(df)
    elif mode == "safety_composite_14d":
        scalar = _reward_safety_composite_14d(df, cfg)
    else:
        raise ValueError(
            "reward_mode must be one of: "
            "'survival_6m', 'survival_14d', 'occode_utility', "
            "'death_or_dependence_6m', 'safety_composite_14d'."
        )

    # Components
    comps: List[pd.Series] = []
    names: List[str] = []
    for name in cfg.reward_components:
        n = name.strip()
        if n == "survival_14d":
            comps.append(_reward_survival_14d(df)); names.append("survival_14d")
        elif n == "survival_6m":
            comps.append(_reward_survival_6m(df)); names.append("survival_6m")
        elif n == "occode_utility":
            comps.append(_reward_occode_utility(df)); names.append("occode_utility")
        elif n == "death_or_dependence_6m":
            comps.append(_reward_death_or_dependence_6m(df)); names.append("death_or_dependence_6m")
        else:
            # treat as indicator numeric column (e.g., H14, TRAN14, MI14, PE14, ISC14)
            if n not in df.columns:
                # keep as all-NaN column so schema is stable
                comps.append(pd.Series(np.nan, index=df.index, dtype=float)); names.append(n)
            else:
                comps.append(pd.to_numeric(df[n], errors="coerce").astype(float)); names.append(n)

    comp_mat = np.vstack([c.to_numpy(dtype="float64") for c in comps]).T if comps else np.zeros((len(df), 0))
    return scalar.to_numpy(dtype="float64"), comp_mat, names


# ----------------------------
# Offline log build + site split
# ----------------------------

def build_offline_log(df: pd.DataFrame, cfg: Optional[ISTLoadConfig] = None) -> Dict:
    cfg = cfg or default_config()

    # Sites
    if cfg.site_col not in df.columns:
        raise KeyError(f"Site column '{cfg.site_col}' not found.")
    sites_num = pd.to_numeric(df[cfg.site_col], errors="coerce").astype(float)
    sites_str = df[cfg.site_col].astype("string")

    # Contexts
    X, feature_names = encode_contexts(df, cfg)

    # Actions
    A_raw, action_names = encode_actions(df, cfg)

    # Rewards
    R_raw, R_comp_raw, comp_names = encode_rewards(df, cfg)

    # Missingness mask
    mask = np.ones(len(df), dtype=bool)

    if cfg.drop_missing_covariates:
        mask &= np.isfinite(X).all(axis=1)

    if cfg.drop_missing_action:
        mask &= np.isfinite(A_raw)

    if cfg.drop_missing_reward:
        mask &= np.isfinite(R_raw)

    mask &= np.isfinite(sites_num.to_numpy(dtype="float64"))

    # Apply mask
    X = X[mask].astype(np.float32)
    A = A_raw[mask].astype(np.int64)
    R = R_raw[mask].astype(np.float32)
    S = sites_num.to_numpy(dtype="float64")[mask].astype(np.int64)
    S_str = sites_str.to_numpy()[mask]
    R_comp = R_comp_raw[mask].astype(np.float32)

    offline_log = {
        "contexts": X,
        "actions": A,
        "rewards": R,
        "reward_components": R_comp,
        "reward_component_names": comp_names,
        "sites": S,
        "site_str": S_str,
        "action_names": action_names,
        "feature_names": feature_names,
        "meta": {
            "action_mode": cfg.action_mode,
            "reward_mode": cfg.reward_mode,
            "covariates": list(cfg.covariates),
            "include_country_feature": cfg.include_country_feature,
            "include_hospital_hash_feature": cfg.include_hospital_hash_feature,
            "hospital_hash_dim": cfg.hospital_hash_dim,
            "site_col": cfg.site_col,
            "n_rows_raw": int(len(df)),
            "n_rows_used": int(mask.sum()),
            "n_sites": int(len(np.unique(S))),
        },
    }
    return offline_log


def split_sites_train_test(
    offline_log: Dict,
    train_site_frac: float = 0.5,
    seed: int = 0,
) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """
    Split by unique site IDs (HOSPNUM).
    """
    if not (0.0 < train_site_frac < 1.0):
        raise ValueError("train_site_frac must be in (0,1).")

    sites = offline_log["sites"]
    unique_sites = np.unique(sites)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_sites)

    n_train = int(round(train_site_frac * len(unique_sites)))
    n_train = max(1, min(n_train, len(unique_sites) - 1))

    train_sites = np.sort(unique_sites[:n_train])
    test_sites = np.sort(unique_sites[n_train:])

    train_mask = np.isin(sites, train_sites)
    test_mask = np.isin(sites, test_sites)

    def sublog(mask: np.ndarray) -> Dict:
        sub = {
            k: (v[mask] if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0] else v)
            for k, v in offline_log.items()
        }
        # meta adjustments
        meta = dict(sub.get("meta", {}))
        meta["n_rows_used"] = int(mask.sum())
        meta["n_sites"] = int(len(np.unique(sub["sites"])))
        sub["meta"] = meta
        return sub

    return sublog(train_mask), sublog(test_mask), train_sites, test_sites


def load_ist_offline_logs(path: str, cfg: Optional[ISTLoadConfig] = None):
    """
    Convenience: read -> build offline_log -> split by sites.
    """
    cfg = cfg or default_config()
    df = read_ist_table(path, cfg)
    log = build_offline_log(df, cfg)
    tr, te, tr_sites, te_sites = split_sites_train_test(log, cfg.train_site_frac, cfg.split_seed)
    return tr, te, tr_sites, te_sites


# ----------------------------
# CLI quick check


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python datasets/ist_loader.py <path_to_IST_corrected.txt>")
        raise SystemExit(2)

    path = sys.argv[1]
    cfg = default_config()
    train_log, test_log, train_sites, test_sites = load_ist_offline_logs(path, cfg)

    print("OK: built IST offline logs")
    print("Action mode:", cfg.action_mode, " (#arms =", len(train_log["action_names"]), ")")
    print("Reward mode:", cfg.reward_mode)
    print("Context dim:", train_log["contexts"].shape[1])
    print("Train rows:", train_log["meta"]["n_rows_used"], "Train sites:", len(train_sites))
    print("Test  rows:", test_log["meta"]["n_rows_used"], "Test sites:", len(test_sites))
    print("First 10 feature names:", train_log["feature_names"][:10])
    print("Action names:", train_log["action_names"])
    print("Reward components:", train_log["reward_component_names"])