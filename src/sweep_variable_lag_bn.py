#!/usr/bin/env python3
"""
Parameter sweep for variable-lag BN causal discovery (mixed data).

This script varies one attribute at a time (e.g., N, T, density, lag distribution,
noise, AR(1) autocorrelation, fraction of binary nodes, confounders, missingness)
and measures metrics of the learner across multiple trials for each setting.

USAGE
---------------
# Sweep sample size T
python3 sweep_variable_lag_bn.py --sweep T --values 500 1000 5000 10000 --trials 5 --N 8 --p_edge 0.15 --L_max 6

# Sweep number of nodes N
python sweep_variable_lag_bn.py --sweep N --values 4 8 16 24 --trials 5 --T 1000 --p_edge 0.15 --L_max 6

# Sweep density (expected indegree via p_edge)
python sweep_variable_lag_bn.py --sweep density --values 0.08 0.15 0.2 0.3 --trials 5 --T 1000 --N 8

# Sweep lag distribution (categorical)
python sweep_variable_lag_bn.py --sweep lagdist --values uniform short long --trials 5 --T 1000 --N 8 --L_max 6

# Sweep noise_std (SNR)
python sweep_variable_lag_bn.py --sweep noise --values 0.4 0.8 1.2 1.6 --trials 5 --T 1000 --N 8

# Sweep AR(1) phi
python sweep_variable_lag_bn.py --sweep phi --values 0.0 0.3 0.6 0.9 --trials 5 --T 1000 --N 8

# Sweep fraction of binary nodes
python sweep_variable_lag_bn.py --sweep fracbin --values 0.2 0.5 0.8 --trials 5 --T 1000 --N 8

# Sweep confounders on/off or count
python sweep_variable_lag_bn.py --sweep conf --values 0 2 4 --trials 5 --T 1000 --N 8

# Sweep missingness level MCAR
python sweep_variable_lag_bn.py --sweep mcar --values 0.0 0.05 0.1 0.2 --trials 5 --T 1000 --N 8

# Sweep MAR missingness strength
python sweep_variable_lag_bn.py --sweep mar --values 0.0 0.05 0.1 0.2 --trials 5 --T 1000 --N 8


Output:
- sweep_table_<sweep>_<timestamp>.csv            (aggregated metrics table)
"""

import argparse
import os
import sys
import csv
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from variable_lag_tabu import compute_score_mixed, tabu_search_greedy_lag
except Exception as e:
    print("ERROR: Could not import 'compute_score_mixed' and 'tabu_search_greedy_lag' "
          "from module 'variable_lag_tabu'. Place that file next to this script.\n\n"
          f"Original import error:\n{e}", file=sys.stderr)
    sys.exit(1)


@dataclass
class DGPConfig:
    N: int
    T: int
    L_max: int
    p_edge: float
    frac_binary: float
    beta_scale: float
    logit_scale: float
    noise_std: float
    phi: float                      # AR(1) autocorrelation for continuous children
    n_confounders: int              # number of latent confounders
    conf_strength: float            # strength of confounder effects
    mcar_rate: float                # MCAR missing proportion
    mar_rate: float                 # MAR missing proportion
    seed: int


def sample_node_types(N: int, frac_binary: float, rng: np.random.Generator) -> Dict[int, str]:
    
    """
    Random node type assignment: returns a dict mapping to 'binary' or 'continuous'
    """

    types = {}
    idx = np.arange(N)
    rng.shuffle(idx)
    n_bin = int(round(frac_binary * N))
    bin_set = set(idx[:n_bin])
    for j in range(N):
        types[j] = 'binary' if j in bin_set else 'continuous'
    return types


def sample_random_dag_with_lags(N: int, p_edge: float, L_max: int, rng: np.random.Generator,
                                lagdist: str = "uniform") -> Dict[int, List[Tuple[int, int]]]:
    
    """
    This function is the Random DAG (acyclic by index order) generator with edge-specific lags.
    lagdist: 'uniform' | 'short' (bias to small lags) | 'long' (bias to large lags)
    """
    
    struct: Dict[int, List[Tuple[int, int]]] = {}
    for v in range(N):
        for u in range(N):
            if u == v:
                continue
            if u < v and rng.random() < p_edge:
                if lagdist == "uniform":
                    lag = int(rng.integers(1, L_max + 1))
                elif lagdist == "short":
                    k = 1
                    while rng.random() > 0.5 and k < L_max:
                        k += 1
                    lag = k
                elif lagdist == "long":
                    lag = int(L_max + 1 - rng.integers(1, L_max + 1))
                    lag = max(1, lag)
                else:
                    lag = int(rng.integers(1, L_max + 1))
                struct.setdefault(v, []).append((u, lag))
    return struct


def sample_coefficients(struct: Dict[int, List[Tuple[int, int]]],
                        node_types: Dict[int, str],
                        beta_scale: float,
                        logit_scale: float,
                        rng: np.random.Generator) -> Dict[Tuple[int, int], float]:
    
    """
    Takes a randomly generated lagged DAG structure and assigns a random weight to every directed edge
    """
    
    coefs: Dict[Tuple[int, int], float] = {}
    for v, parents in struct.items():
        for (u, _lag) in parents:
            scale = logit_scale if node_types[v] == 'binary' else beta_scale
            coefs[(u, v)] = float(rng.normal(0.0, scale))
    return coefs


def simulate_time_series(T: int,
                         struct: Dict[int, List[Tuple[int, int]]],
                         node_types: Dict[int, str],
                         coefs: Dict[Tuple[int, int], float],
                         noise_std: float,
                         rng: np.random.Generator,
                         phi: float = 0.0,
                         n_confounders: int = 0,
                         conf_strength: float = 0.5) -> np.ndarray:
    
    """
    Generates a synthetic multivariate time series from a lagged causal graph: at each time step it computes each node's value as a weighted sum of its parents' past values (with edge-specific lags), 
    optionally adds latent confounder signals, and then produces either a Gaussian continuous value (with optional AR(1) self-dependence and noise) or a binary value via a logistic sigmoid. 
    It returns the resulting T X N data matrix.
    """

    N = len(node_types)
    data = np.zeros((T, N), dtype=float)
    data[:5, :] = rng.normal(0.0, 1.0, size=(5, N))

    # Confounders
    conf_pairs: List[Tuple[int, int, int]] = []
    U = None
    if n_confounders > 0:
        U = np.zeros((T, n_confounders), dtype=float)
        cphi = min(max(phi/2.0, 0.0), 0.99)
        all_pairs = [(i, j) for i in range(N) for j in range(N) if i < j]
        rng.shuffle(all_pairs)
        for k in range(n_confounders):
            a, b = all_pairs[k % len(all_pairs)]
            lag_k = int(rng.integers(1, 4))
            conf_pairs.append((a, b, lag_k))
            U[0, k] = rng.normal(0.0, 1.0)
            for t in range(1, T):
                U[t, k] = cphi * U[t-1, k] + rng.normal(0.0, 1.0)

    def sigmoid(x):

        """
        Computes the logistic sigmoid for a scalar or NumPy array, returning a float for scalar input and an array otherwise.
        """

        x_arr = np.asarray(x, dtype=float)
        scalar_input = (x_arr.ndim == 0)
        if scalar_input:
            x_arr = x_arr.reshape(1)
        out = np.empty_like(x_arr, dtype=float)
        mask = x_arr >= 0
        out[mask] = 1.0 / (1.0 + np.exp(-x_arr[mask]))
        xm = x_arr[~mask]
        ex = np.exp(xm)
        out[~mask] = ex / (1.0 + ex)
        return float(out[0]) if scalar_input else out

    for t in range(T):
        for v in range(N):
            parents = struct.get(v, [])
            linear = 0.0
            for (u, lag) in parents:
                tp = t - lag
                if tp >= 0:
                    linear += coefs[(u, v)] * data[tp, u]

            if n_confounders > 0:
                for k, (a, b, lag_k) in enumerate(conf_pairs):
                    tp = t - lag_k
                    if tp >= 0 and (v == a or v == b):
                        linear += conf_strength * U[tp, k]

            if node_types[v] == 'continuous':
                ar_term = (phi * data[t-1, v]) if (t > 0 and phi != 0.0) else 0.0
                val = ar_term + linear + rng.normal(0.0, noise_std)
                data[t, v] = val
            else:
                p = sigmoid(linear)
                p = np.clip(p, 1e-6, 1 - 1e-6)
                data[t, v] = rng.binomial(1, p)

    return data


def edge_set(struct: Dict[int, List[Tuple[int, int]]]) -> set:
    
    """
    Takes the lagged parent structure dict and converts it into a set of directed edges.
    """
    es = set()
    for v, parents in struct.items():
        for (u, _lag) in parents:
            es.add((u, v))
    return es


def edge_lag_map(struct: Dict[int, List[Tuple[int, int]]]) -> Dict[Tuple[int, int], int]:

    """
    Converts the lagged parent-list structure into a dictionary mapping each directed edge to its lag value.
    """

    mp = {}
    for v, parents in struct.items():
        for (u, lag) in parents:
            mp[(u, v)] = int(lag)
    return mp


def augment_truth_with_ar(true_edges: set,
                          true_lags: Dict[Tuple[int, int], int],
                          node_types: Dict[int, str],
                          phi: float) -> None:

    """Augment ground-truth edges/lags with AR(1) self-lag links.

    The DGP for continuous nodes includes an AR(1) term of the form
        X_{t} += phi * X_{t-1}
    which corresponds to an autoregressive edge X_{t-1} -> X_{t}.

    In our compact (u,v) representation (without explicit time indices), we
    represent this as a self-edge (v,v) with lag 1.
    """

    if phi == 0.0:
        return
    for v, t in node_types.items():
        if t == 'continuous':
            true_edges.add((v, v))
            true_lags[(v, v)] = 1


def precision_recall_f1(true_edges: set, pred_edges: set):

    """
    Computes precision, recall, and F1 score.
    """

    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def shd(true_edges: set, pred_edges: set) -> int:

    """
    Computes Structural Hamming Distance (SHD) by counting how many edges are extra (adds), 
    missing (deletes), or predicted in the wrong direction (reversals) between the predicted and true edge sets.
    """
    
    # We allow autoregressive edges represented as (u,u).
    # A "reversal" is only meaningful for u != v. The classic implementation
    # based on set reversal would incorrectly count shared self-edges as reversals
    # because (u,u) is its own reverse.

    rev = 0
    add = 0
    delete = 0

    for (u, v) in true_edges:
        if u == v:
            if (u, v) not in pred_edges:
                delete += 1
        else:
            if (u, v) in pred_edges:
                continue
            if (v, u) in pred_edges:
                rev += 1
            else:
                delete += 1

    for (u, v) in pred_edges:
        if u == v:
            if (u, v) not in true_edges:
                add += 1
        else:
            if (u, v) in true_edges:
                continue
            if (v, u) in true_edges:
                continue  # this case is already counted as a reversal above
            add += 1

    return add + delete + rev


def lag_mae(true_lags: Dict[Tuple[int, int], int], pred_lags: Dict[Tuple[int, int], int]) -> float:

    """
    Computes the average absolute difference between true and predicted lags, but only for edges that exist in both graphs.
    """

    common = set(true_lags.keys()) & set(pred_lags.keys())
    if not common:
        return float('nan')
    err = [abs(true_lags[e] - pred_lags[e]) for e in common]
    return float(np.mean(err))

def bsf(true_edges: set, pred_edges: set, N: int) -> float:
   
    """
    It computes a balanced score of how well the predicted directed edges match the true graph by rewarding correct edges and correct non-edges and penalizing false positives and false negatives, 
    normalized by the number of true edges and non-edges.
    """
   
    # Universe of possible directed edges.
    # We include self-edges (u,u)
    all_pairs = {(u, v) for u in range(N) for v in range(N)}

    E = set(true_edges)       # true edges
    M = all_pairs - E         # true non-edges

    TP = len(E & pred_edges)
    FP = len(pred_edges & M)
    FN = len(E - pred_edges)
    TN = len(M - pred_edges)

    if len(E) == 0 or len(M) == 0:
        return float('nan')   

    return 0.5 * ((TP/len(E)) + (TN/len(M)) - (FP/len(M)) - (FN/len(E)))



def apply_missingness_MCAR(data: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:

    """
    Defines a function that injects missing values into data using MCAR.
    """

    if rate <= 0:
        return data
    mask = rng.random(size=data.shape) < rate
    X = data.copy()
    X[mask] = np.nan
    return X


def apply_missingness_MAR(data: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    
    """
    This function picks one variable as a “driver” and makes rows more missing when that driver's value is large.
    """
    
    if rate <= 0:
        return data
    X = data.copy()
    T, N = X.shape
    driver = int(rng.integers(0, N))
    z = np.abs(X[:, driver])
    z = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z) + 1e-9)
    base = rate / 2.0
    prob = np.clip(base + z * (rate - base), 0.0, 0.9)
    M = rng.random(size=X.shape)
    miss = (M < prob[:, None])
    X[miss] = np.nan
    return X


def simple_impute(X: np.ndarray) -> np.ndarray:

    """
    Takes a NumPy array X and returns an imputed NumPy array.
    """

    Xi = X.copy()
    T, N = Xi.shape
    for j in range(N):
        col = Xi[:, j]
        if np.isnan(col).any():
            finite = col[~np.isnan(col)]
            uniq = np.unique(finite)
            if uniq.size <= 2 and set(uniq).issubset({0.0, 1.0}):
                ones = np.nansum(col)
                zeros = np.sum(~np.isnan(col)) - ones
                fill = 1.0 if ones >= zeros else 0.0
            else:
                fill = float(np.nanmean(col)) if np.isfinite(np.nanmean(col)) else 0.0
            col[np.isnan(col)] = fill
            Xi[:, j] = col
    return Xi


def run_one_setting(valuestr: str, base_cfg: DGPConfig, sweep: str, learner_kwargs: dict,
                    lagdist: str) -> dict:
    
    """
    For one sweep value, runs multiple trials, generates synthetic data each time, runs the learner, computes metrics, then returns per-trial and averaged results.
    """
    
    rng = np.random.default_rng(base_cfg.seed)
    trials = learner_kwargs.pop('_trials')
    results = []

    for _ in range(trials):
        cfg = DGPConfig(**vars(base_cfg))

        if sweep == 'T':
            cfg.T = int(valuestr)
        elif sweep == 'N':
            cfg.N = int(valuestr)
        elif sweep == 'density':
            cfg.p_edge = float(valuestr)
        elif sweep == 'lagdist':
            pass
        elif sweep == 'noise':
            cfg.noise_std = float(valuestr)
        elif sweep == 'phi':
            cfg.phi = float(valuestr)
        elif sweep == 'fracbin':
            cfg.frac_binary = float(valuestr)
        elif sweep == 'conf':
            cfg.n_confounders = int(valuestr)
        elif sweep == 'mcar':
            cfg.mcar_rate = float(valuestr)
        elif sweep == 'mar':
            cfg.mar_rate = float(valuestr)

        trial_seed = int(rng.integers(0, 2**31 - 1))
        cfg.seed = trial_seed

        rng_trial = np.random.default_rng(cfg.seed)
        node_types = sample_node_types(cfg.N, cfg.frac_binary, rng_trial)
        true_struct = sample_random_dag_with_lags(cfg.N, cfg.p_edge, cfg.L_max, rng_trial, lagdist=lagdist)
        coefs = sample_coefficients(true_struct, node_types, cfg.beta_scale, cfg.logit_scale, rng_trial)

        data = simulate_time_series(cfg.T, true_struct, node_types, coefs, cfg.noise_std, rng_trial,
                                    phi=cfg.phi, n_confounders=cfg.n_confounders, conf_strength=cfg.conf_strength)

        if cfg.mcar_rate > 0.0:
            data = apply_missingness_MCAR(data, cfg.mcar_rate, rng_trial)
        if cfg.mar_rate > 0.0:
            data = apply_missingness_MAR(data, cfg.mar_rate, rng_trial)

        if (cfg.mcar_rate > 0.0) or (cfg.mar_rate > 0.0):
            data = simple_impute(data)

        learned_struct, best_score = tabu_search_greedy_lag(
            data,
            node_types,
            max_iter=learner_kwargs.get('max_iter', 100),
            tabu_length=learner_kwargs.get('tabu_length', 10),
            penalty_coef=learner_kwargs.get('penalty_coef', 0.5),
            L_max=learner_kwargs.get('L_max', cfg.L_max),
            seed=learner_kwargs.get('seed', cfg.seed + 12345)
        )

        te = edge_set(true_struct)
        tl = edge_lag_map(true_struct)
        augment_truth_with_ar(te, tl, node_types, cfg.phi)

        pe = edge_set(learned_struct)
        prec, rec, f1 = precision_recall_f1(te, pe)
        d = shd(te, pe)
        pl = edge_lag_map(learned_struct)
        mae = lag_mae(tl, pl)
        bsf_val = bsf(te, pe, cfg.N)

        results.append(dict(
            precision=prec, recall=rec, f1=f1, shd=d, lag_mae=mae, bsf=bsf_val,
            best_score=best_score
        ))

    def mean_ignore_nan(key):

        """
        It collects all the trial values for a given metric, throws away any that are NaN, and returns their average
        """
        vals = [r[key] for r in results if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float('nan')

    agg = {
        'precision': mean_ignore_nan('precision'),
        'recall': mean_ignore_nan('recall'),
        'f1': mean_ignore_nan('f1'),
        'shd': mean_ignore_nan('shd'),
        'lag_mae': mean_ignore_nan('lag_mae'),
        'bsf': mean_ignore_nan('bsf'),  
        'best_score': mean_ignore_nan('best_score'),
    }
    return {'agg': agg, 'results': results}


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for variable-lag BN (mixed data)")
    parser.add_argument('--sweep', type=str, required=True,
                        choices=['T','N','density','lagdist','noise','phi','fracbin','conf','mcar','mar'])
    parser.add_argument('--values', nargs='+', required=True, help="Values to sweep over")
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    # DGP base defaults
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--T', type=int, default=400)
    parser.add_argument('--L_max', type=int, default=6)
    parser.add_argument('--p_edge', type=float, default=0.15)
    parser.add_argument('--frac_binary', type=float, default=0.5)
    parser.add_argument('--beta_scale', type=float, default=0.8)
    parser.add_argument('--logit_scale', type=float, default=1.2)
    parser.add_argument('--noise_std', type=float, default=0.8)
    parser.add_argument('--phi', type=float, default=0.0)
    parser.add_argument('--n_confounders', type=int, default=0)
    parser.add_argument('--conf_strength', type=float, default=0.5)
    parser.add_argument('--mcar_rate', type=float, default=0.0)
    parser.add_argument('--mar_rate', type=float, default=0.0)

    # learner defaults
    parser.add_argument('--penalty_coef', type=float, default=0.5)
    parser.add_argument('--tabu_length', type=int, default=10)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--module_L_max', type=int, default=None)

    parser.add_argument('--lagdist', type=str, default='uniform', choices=['uniform','short','long'],
                        help="Lag distribution when not sweeping 'lagdist'.")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ts = int(time.time())

    base_cfg = DGPConfig(
        N=args.N, T=args.T, L_max=args.L_max, p_edge=args.p_edge, frac_binary=args.frac_binary,
        beta_scale=args.beta_scale, logit_scale=args.logit_scale, noise_std=args.noise_std,
        phi=args.phi, n_confounders=args.n_confounders, conf_strength=args.conf_strength,
        mcar_rate=args.mcar_rate, mar_rate=args.mar_rate, seed=int(rng.integers(0, 2**31 - 1))
    )

    learner_kwargs = dict(
        max_iter=args.max_iter,
        tabu_length=args.tabu_length,
        penalty_coef=args.penalty_coef,
        L_max=(args.module_L_max if args.module_L_max is not None else args.L_max),
        seed=int(rng.integers(0, 2**31 - 1)),
        _trials=args.trials
    )

    rows = []
    for valuestr in args.values:
        lagdist = args.lagdist
        if args.sweep == 'lagdist':
            lagdist = valuestr

        out = run_one_setting(valuestr, base_cfg, args.sweep, learner_kwargs.copy(), lagdist)
        agg = out['agg']
        rows.append((valuestr, agg['precision'], agg['recall'], agg['f1'], agg['bsf'],
                     agg['shd'], agg['lag_mae'], agg['best_score']))

        print(f"[{args.sweep}={valuestr}] "
              f"P={agg['precision']:.3f} R={agg['recall']:.3f} F1={agg['f1']:.3f} "
              f"BSF={agg['bsf']:.3f} SHD={agg['shd']:.2f} LagMAE={agg['lag_mae']:.3f} "
              f"BestScore={agg['best_score']:.1f}")

    csv_path = f"sweep_table_{args.sweep}_{ts}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['value','precision','recall','f1','bsf','shd','lag_mae','best_score'])
        for r in rows:
            w.writerow(r)

    print("\nSaved artifacts:")
    print(f" - {csv_path}")

if __name__ == '__main__':
    main()
