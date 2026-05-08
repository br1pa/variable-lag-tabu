"""
Run the ORIGINAL variable-lag tabu search (variable_lag_tabu.py) on `data.csv`,
then compute summary metrics and a policy-evaluation table.

Outputs (written to --outdir)
-----------------------------
- edges.csv    : learned edges (parent, child, lag)
- policy.csv   : policy evaluation 
- metrics.json : summary metrics (edges, free params, LL, BIC, score, lag stats)

python3 real_world_application.py \
  --csv "data.csv" \
  --outdir "results/real_world_$(date +%Y%m%d_%H%M%S)" \
  --L_max 6 --max_iter 100 --tabu_length 10 --penalty_coef 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from variable_lag_tabu import tabu_search_greedy_lag, compute_score_mixed, _logistic_irls  # type: ignore


Parent = Tuple[int, int] 


# =============================================================================
# Preprocessing helpers
# =============================================================================

def _encode_ordered_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def enc(cols, order):
        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            if col in out.columns:
                out[col] = pd.Series(
                    pd.Categorical(out[col], categories=order, ordered=True).codes,
                    index=out.index,
                ).replace(-1, np.nan)
                return

    enc(["Schools"], ["Closed", "Partially open", "Open"])
    enc(["Face masks", "Face.masks"], ["Yes", "Optional", "No"])
    enc(
        ["Lockdown severity", "Lockdown.severity"],
        ["Severe_lockdown", "Moderate_lockdown", "Weak_lockdown",
         "Social_distancing", "No_or_limited_measures"],
    )
    enc(
        [
            "Majority COVID-19 variant [before 12/05 used https://ourworldindata.org/covid-vaccinations with 'VARIANTS' as metric]",
            "Majority COVID-19 variant",
            "Majority.COVID.19.variant",
        ],
        ["Initial", "Alpha", "Delta", "Omicron", "Omicron BA.2"],
    )
    enc(["Season"], ["Winter", "Spring", "Summer", "Autumn"])
    return out

def infer_node_types(df_numeric: pd.DataFrame, var_names: List[str]) -> Dict[int, str]:
    """
      - if a column has <=2 unique values, mark as 'binary'
      - else mark as 'continuous'
    """
    node_types: Dict[int, str] = {}
    for i, name in enumerate(var_names):
        vals = df_numeric[name].to_numpy()
        uniq = np.unique(vals[~np.isnan(vals)])
        node_types[i] = "binary" if uniq.size <= 2 else "continuous"
    return node_types


# =============================================================================
# Metrics (LL, BIC, free params, lag stats)
# =============================================================================

def _fit_local_linear_gaussian(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = y.shape[0]
    rss = float(np.sum(resid ** 2))
    sigma2 = max(rss / max(n, 1), 1e-12)
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
    return beta, float(ll)


def compute_metrics(
    data: np.ndarray,
    structure: Dict[int, List[Parent]],
    node_types: Dict[int, str],
    penalty_coef: float,
    L_max: int,
) -> Dict[str, float]:
    """
    Compute:
      - No_of_edges
      - No_of_free_parameters (intercept + slopes per node)
      - LL (sum over local models)
      - BIC (sum over local: ll - 0.5*p*log(n_eff))
      - score_used_by_algorithm (compute_score_mixed output)
      - lag penalty stats
    """
    T, n_vars = data.shape
    ll_total = 0.0
    free_params = 0
    bic_total = 0.0

    lags: List[int] = []
    lag_penalty = 0

    for j in range(n_vars):
        parents = structure.get(j, [])
        if parents:
            parents = [(p, min(int(lag), L_max)) for (p, lag) in parents]
        Lj = max([lag for (_, lag) in parents], default=0)
        n_eff = T - Lj
        if n_eff <= 1:
            return {
                "No_of_edges": float(sum(len(v) for v in structure.values())),
                "No_of_free_parameters": float("nan"),
                "Log_likelihood_LL": float("-inf"),
                "BIC": float("-inf"),
                "score_used_by_algorithm": float("-inf"),
                "lag_penalty_sum": float("nan"),
                "avg_lag": float("nan"),
                "max_lag": float("nan"),
            }

        y_raw = data[Lj:, j]
        X_cols = []
        for (p, lag) in parents:
            p_series = data[Lj - lag : T - lag, p]
            if node_types.get(p, "continuous") == "binary":
                uniq = np.unique(p_series)
                if uniq.size == 1:
                    col = (p_series == uniq[0]).astype(float)
                elif uniq.size == 2:
                    col = (p_series == uniq.max()).astype(float)
                else:
                    col = p_series.astype(float)
                X_cols.append(col.reshape(-1, 1))
            else:
                X_cols.append(p_series.reshape(-1, 1).astype(float))

        if X_cols:
            X = np.hstack(X_cols)
            X = np.hstack([np.ones((n_eff, 1)), X])  # intercept
        else:
            X = np.ones((n_eff, 1))

        child_type = node_types.get(j, "continuous")

        if child_type == "continuous":
            _, ll = _fit_local_linear_gaussian(X, y_raw.astype(float))
            p_j = int(X.shape[1])
        else:
            uniq_y = np.unique(y_raw)
            if uniq_y.size <= 2:
                y = (y_raw == uniq_y.max()).astype(float)
                _, ll = _logistic_irls(X.astype(float), y.astype(float))
                p_j = int(X.shape[1])
            else:
                _, ll = _fit_local_linear_gaussian(X, y_raw.astype(float))
                p_j = int(X.shape[1])
            
        ll_total += float(ll)
        free_params += p_j
        bic_total += float(ll) - 0.5 * p_j * np.log(max(n_eff, 2))

    for child, parents in structure.items():
        for (_, lag) in parents:
            lag = min(int(lag), L_max)
            lags.append(lag)
            if lag > 1:
                lag_penalty += (lag - 1)

    n_edges = int(sum(len(v) for v in structure.values()))
    avg_lag = float(np.mean(lags)) if lags else 0.0
    max_lag = int(np.max(lags)) if lags else 0

    score_used = float(compute_score_mixed(data, structure, node_types, penalty_coef=penalty_coef, L_max=L_max))

    return {
        "No_of_edges": float(n_edges),
        "No_of_free_parameters": float(free_params),
        "Log_likelihood_LL": float(ll_total),
        "BIC": float(bic_total),
        "score_used_by_algorithm": float(score_used),
        "lag_penalty_sum": float(lag_penalty),
        "avg_lag": float(avg_lag),
        "max_lag": float(max_lag),
    }


# =============================================================================
# Policy evaluation (Table 7 replication)
# =============================================================================

def _kmeans_centroids_3(x: np.ndarray, seed: int = 0) -> Tuple[float, float, float]:
    """1D k-means (k=3), returns (low, mid, high) centroids."""
    x = x.astype(float)
    c = np.quantile(x, [0.2, 0.5, 0.8]).astype(float)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        d = np.abs(x[:, None] - c[None, :])
        a = np.argmin(d, axis=1)
        new_c = c.copy()
        for k in range(3):
            if np.any(a == k):
                new_c[k] = float(np.mean(x[a == k]))
            else:
                new_c[k] = float(rng.choice(x))
        if np.max(np.abs(new_c - c)) < 1e-6:
            c = new_c
            break
        c = new_c
    c = np.sort(c)
    return float(c[0]), float(c[1]), float(c[2])


def policy_evaluation_linear(
    df_numeric: pd.DataFrame,
    structure: Dict[int, List[Parent]],
    var_names: List[str],
    L_max: int,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Table 7 analogue:
      - Causal effect identified if there is a direct edge Cause(t-1)->Effect(t), i.e. lag == 1.
      - Direction matches:
          if Y_t = a + b*X_{t-1} + ..., then decreasing X should decrease Y.
    """
    def squash(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    name_map = {squash(n): n for n in var_names}

    def find(name: str) -> Optional[str]:
        return name_map.get(squash(name))

    mobility_candidates = [
        "Flights (7-day moving average)",
        "OpenTable restaurant bookings (London) index",
        "Google homeworking (Greater London) mobility index",
        "Google workplace (Greater London) mobility index",
        "Apple walking (London) mobility index",
        "Google parks (Greater London) mobility index",
        "Google retail & recreation (Greater London) mobility index",
        "Google grocery & pharmacy (Greater London) mobility index",
        "Google transit stations mobility index",
        "TfL Tube mobility index",
        "TfL Bus mobility index",
        "Citymapper journeys mobility index",
    ]
    infection_candidates = [
        "New cases",
        "New infections",
        "Reinfections",
        "New cases (specimen date)",
    ]

    mobility = [c for c in (find(x) for x in mobility_candidates) if c is not None]
    infection = [c for c in (find(x) for x in infection_candidates) if c is not None]

    name_to_idx = {n: i for i, n in enumerate(var_names)}
    data = df_numeric[var_names].to_numpy(dtype=float)

    # Extract beta for lag-1 edges via OLS for each affected child.
    coeffs: Dict[Tuple[int, int, int], float] = {}
    T = data.shape[0]

    for child, parents in structure.items():
        parents = [(p, lag) for (p, lag) in parents if 1 <= lag <= L_max]
        if not parents:
            continue
        Lj = max(lag for _, lag in parents)
        n_eff = T - Lj
        if n_eff <= 2:
            continue

        y = data[Lj:, child]
        X_cols = []
        meta = []
        for p, lag in parents:
            X_cols.append(data[Lj - lag : T - lag, p].reshape(-1, 1))
            meta.append((p, child, lag))
        X = np.hstack(X_cols)
        X = np.hstack([np.ones((n_eff, 1)), X])

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        for k, (p, ch, lag) in enumerate(meta, start=1):
            coeffs[(p, ch, lag)] = float(beta[k])

    # Expected direction adjustment:
    expected_dir = {m: +1 for m in mobility}
    for m in mobility:
        if "homeworking" in m.lower():
            expected_dir[m] = -1

    rows = []
    for x in mobility:
        for y in infection:
            xi = name_to_idx[x]
            yi = name_to_idx[y]
            beta = coeffs.get((xi, yi, 1))
            identified = beta is not None

            ace = None
            dir_match = False
            if identified:
                x_lag = data[:-1, xi]  
                low, mid, high = _kmeans_centroids_3(x_lag, seed=seed)
                ace_raw = 0.5 * (beta * (mid - high) + beta * (low - high))
                ace = float(ace_raw * expected_dir[x])
                dir_match = ace < 0

            rows.append(
                {
                    "Cause": x,
                    "Effect": y,
                    "Identified (edge lag1)": int(identified),
                    "beta (lag1)": None if beta is None else float(beta),
                    "ACE approx (Eq4 analogue)": ace,
                    "Direction matches knowledge?": int(dir_match),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Lag distribution plot
# =============================================================================

def plot_lag_distribution(edges_df: pd.DataFrame, out_png: Path, out_csv: Path, L_max: int) -> Dict[int, int]:
    """
    Save a bar chart showing how many learned edges have lag 1, lag 2, ..., lag L_max.
    Also saves the underlying counts to CSV.

    Returns a dict {lag: count}.
    """
    if edges_df.empty or "lag" not in edges_df.columns:
        counts = {lag: 0 for lag in range(1, int(L_max) + 1)}
    else:
        vc = edges_df["lag"].astype(int).value_counts().to_dict()
        counts = {lag: int(vc.get(lag, 0)) for lag in range(1, int(L_max) + 1)}

    # Save counts CSV
    pd.DataFrame({"lag": list(counts.keys()), "count": list(counts.values())}).to_csv(out_csv, index=False)

    # Plot
    lags = list(counts.keys())
    vals = [counts[l] for l in lags]

    fig = plt.figure()
    plt.bar(lags, vals)
    plt.xlabel("Lag")
    plt.ylabel("Number of edges")
    plt.title("Lag distribution in learnt structure")
    plt.xticks(lags)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return counts


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data.csv", help="Path to the CSV data")
    ap.add_argument("--outdir", type=str, default="out_variable_lag_tabu_original", help="Output directory")
    ap.add_argument("--L_max", type=int, default=5, help="Maximum lag allowed")
    ap.add_argument("--max_iter", type=int, default=100, help="Tabu iterations after greedy climb")
    ap.add_argument("--tabu_length", type=int, default=10, help="Tabu list length")
    ap.add_argument("--penalty_coef", type=float, default=0.5, help="Lag penalty coefficient")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument(
        "--max_vars",
        type=int,
        default=0,
        help="Optional: limit to first N variables (0 = use all). Helpful to reduce runtime.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})

    df_enc = _encode_ordered_categories(df)

    # data.csv is complete, but for robustness
    df_imp = df_enc.dropna(axis=0, how="any").reset_index(drop=True)

    var_names = [c for c in df_imp.columns if c.lower() != "date"]
    if args.max_vars and args.max_vars > 0:
        var_names = var_names[: int(args.max_vars)]

    data = df_imp[var_names].to_numpy(dtype=float)
    node_types = infer_node_types(df_imp, var_names)

    best_struct, best_score = tabu_search_greedy_lag(
        data=data,
        node_types=node_types,
        max_iter=int(args.max_iter),
        tabu_length=int(args.tabu_length),
        penalty_coef=float(args.penalty_coef),
        L_max=int(args.L_max),
        seed=int(args.seed),
    )

    # Edges CSV
    edges = []
    for child, parents in best_struct.items():
        for p, lag in parents:
            edges.append(
                {
                    "parent_idx": int(p),
                    "child_idx": int(child),
                    "lag": int(lag),
                    "parent": var_names[int(p)],
                    "child": var_names[int(child)],
                }
            )
    edges_df = pd.DataFrame(edges)
    if edges_df.empty:
        edges_df = pd.DataFrame(columns=["parent_idx", "child_idx", "lag", "parent", "child"])
    else:
        edges_df = edges_df.sort_values(["child", "parent", "lag"])
    edges_df.to_csv(outdir / "edges.csv", index=False)

    # Lag distribution plot + CSV
    lag_png = outdir / "lag_distribution.png"
    lag_csv = outdir / "lag_distribution.csv"
    lag_counts = plot_lag_distribution(edges_df, lag_png, lag_csv, L_max=int(args.L_max))


    # Metrics JSON
    metrics_tbl6 = compute_metrics(
        data=data,
        structure=best_struct,
        node_types=node_types,
        penalty_coef=float(args.penalty_coef),
        L_max=int(args.L_max),
    )

    # Policy evaluation CSV + summary
    policy_df = policy_evaluation_linear(
        df_numeric=df_imp[var_names],
        structure=best_struct,
        var_names=var_names,
        L_max=int(args.L_max),
        seed=int(args.seed),
    )
    policy_df.to_csv(outdir / "policy.csv", index=False)

    identified = int(policy_df["Identified (edge lag1)"].sum()) if not policy_df.empty else 0
    dir_matches = int(policy_df["Direction matches knowledge?"].sum()) if not policy_df.empty else 0

    metrics = {
        "algorithm": "variable_lag_tabu.py :: tabu_search_greedy_lag (original)",
        "n_rows_used": int(len(df_imp)),
        "n_variables": int(len(var_names)),
        "hyperparameters": {
            "L_max": int(args.L_max),
            "max_iter": int(args.max_iter),
            "tabu_length": int(args.tabu_length),
            "penalty_coef": float(args.penalty_coef),
            "seed": int(args.seed),
            "max_vars": int(args.max_vars),
        },
        "graphical_modelling_metrics_(Table_6_analogue)": {
            **metrics_tbl6,
            "best_score_returned_by_search": float(best_score),
            "lag_distribution_counts_by_lag": {str(k): int(v) for k, v in lag_counts.items()},
        },
        "policy_evaluation_metrics_(Table_7_analogue)": {
            "No_of_causal_effects_identified_(lag1_edges_interactions_to_infections)": int(identified),
            "No_of_times_direction_matches_knowledge_(ACE<0_after_adjustments)": int(dir_matches),
            "max_possible_effects_(len(mobility)*len(infection))": int(len(policy_df)),
        },
        "files_written": {
            "edges": str((outdir / "edges.csv").resolve()),
            "policy": str((outdir / "policy.csv").resolve()),
            "metrics": str((outdir / "metrics.json").resolve()),
            "lag_distribution_plot": str((outdir / "lag_distribution.png").resolve()),
            "lag_distribution_counts": str((outdir / "lag_distribution.csv").resolve()),
        },
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
