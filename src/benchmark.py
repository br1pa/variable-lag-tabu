#!/usr/bin/env python3
"""Reproducible benchmark for the variable-lag mixed Bayesian network.

This script runs the variable lag tabu method and a same-score fixed-lag ablation
on all eligible synthetic settings, and optionally runs external time-series
causal-discovery methods on settings that match their assumptions:

* PCMCI (Tigramite)
* DYNOTEARS (CausalNex)
* LPCMCI (Tigramite)
* VARLiNGAM (lingam; non-Gaussian continuous settings only by default)

The output distinguishes variable-level directed adjacency recovery from exact
lagged-edge recovery.  In particular, it treats (parent, child, lag) as the
primary exact-recovery target and reports lag MAE only conditionally on a
correctly recovered ordered adjacency.

Examples
--------
python benchmark.py   --suite multilag   --trials 20   --methods proposed fixed_lag_1 fixed_lag_cv pcmci dynotears lpcmci   --lpcmci-all-continuous   --no-lpcmci-include-contemporaneous --lpcmci-max-p-global 3 --lpcmci-max-p-non-ancestral 3 --lpcmci-max-q-global 100 --learner-file variable_lag_tabu.py   --parallel-learner-file variable_lag_tabu_parallel.py   --proposed-parallel   --proposed-jobs 8   --multi-lag-fraction 0.5   --lags-per-multi-edge 2   --output-dir benchmark_multilag_results --seed 42
python benchmark.py   --suite reviewer   --trials 20   --seed 42   --methods proposed   --proposed-parallel   --parallel-learner-file variable_lag_tabu_parallel.py   --proposed-jobs 4   --output-dir proposed_parallel
python benchmark.py --suite reviewer --trials 20  --seed 42 --methods proposed fixed_lag_1 fixed_lag_cv pcmci lpcmci dynotears varlingam --lpcmci-all-continuous --no-lpcmci-include-contemporaneous --lpcmci-max-p-global 3 --lpcmci-max-p-non-ancestral 3 --lpcmci-max-q-global 100 --output-dir proposed_serial

Optional packages
-----------------
PCMCI and LPCMCI require ``tigramite``.  VARLiNGAM requires ``lingam``.
DYNOTEARS uses ``causalnex`` and is most reliably installed in the Python 3.10
environment recommended by CausalNex.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pickle
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


Structure = Dict[int, List[Tuple[int, int]]]
LaggedEdge = Tuple[int, int, int]
DirectedPair = Tuple[int, int]
UndirectedPair = Tuple[int, int]

ALL_METHODS = (
    "proposed",
    "fixed_lag_1",
    "fixed_lag_cv",
    "pcmci",
    "dynotears",
    "lpcmci",
    "varlingam",
    "td2c",
)

PRIMARY_METRICS = (
    "adjacency_precision",
    "adjacency_recall",
    "adjacency_f1",
    "adjacency_shd",
    "adjacency_bsf",
    "exact_precision",
    "exact_recall",
    "exact_f1",
    "exact_shd",
    "lag_mae_conditional",
    "multi_lag_edge_recall",
    "multi_lag_pair_any_recall",
    "multi_lag_pair_exact_set_accuracy",
    "runtime_seconds",
)


@dataclass(frozen=True)
class DGPConfig:
    N: int = 8
    T: int = 1000
    L_max: int = 6
    p_edge: float = 0.15
    frac_binary: float = 0.5
    beta_scale: float = 0.8
    logit_scale: float = 1.2
    noise_std: float = 0.8
    noise_distribution: str = "gaussian"
    phi: float = 0.0
    n_confounders: int = 0
    conf_strength: float = 0.5
    mcar_rate: float = 0.0
    mar_rate: float = 0.0
    multi_lag_fraction: float = 0.0
    lags_per_multi_edge: int = 2



@dataclass
class SimulatedTrial:
    data: np.ndarray
    node_types: Dict[int, str]
    true_structure: Structure
    true_edges: Set[LaggedEdge]
    confounded_pairs: Set[UndirectedPair]


@dataclass
class MethodOutput:
    lagged_edges: Set[LaggedEdge]
    contemporaneous_edges: Set[DirectedPair]
    score: float = float("nan")
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class SkipMethod(RuntimeError):
    """Raised when a requested method is inapplicable or unavailable."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    nonnegative = z >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-z[nonnegative]))
    ez = np.exp(z[~nonnegative])
    out[~nonnegative] = ez / (1.0 + ez)
    return out


def _log1pexp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _logistic_irls(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    ridge: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    n_features = X.shape[1]
    beta = np.zeros(n_features, dtype=float)
    for _ in range(max_iter):
        eta = X @ beta
        mu = _sigmoid(eta)
        weights = np.clip(mu * (1.0 - mu), 1e-9, None)
        z = eta + (y - mu) / weights
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, None]
        zw = z * sqrt_w
        lhs = Xw.T @ Xw + ridge * np.eye(n_features)
        rhs = Xw.T @ zw
        try:
            beta_new = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    loglik = float(np.sum(y * eta - _log1pexp(eta)))
    return beta, loglik


def _binary01(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values)
    unique = np.unique(values)
    if unique.size == 0:
        raise ValueError(f"{label} has no observations")
    if set(unique.tolist()).issubset({0.0, 1.0}):
        return values.astype(float)
    if unique.size <= 2:
        return (values == unique.max()).astype(float)
    raise ValueError(f"{label} was declared binary but has {unique.size} levels")


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------


def sample_node_types(
    n_nodes: int, frac_binary: float, rng: np.random.Generator
) -> Dict[int, str]:
    if not 0.0 <= frac_binary <= 1.0:
        raise ValueError("frac_binary must lie in [0, 1]")
    order = np.arange(n_nodes)
    rng.shuffle(order)
    n_binary = int(round(frac_binary * n_nodes))
    binary_nodes = set(int(x) for x in order[:n_binary])
    return {
        node: ("binary" if node in binary_nodes else "continuous")
        for node in range(n_nodes)
    }


def sample_lag(
    L_max: int, lag_distribution: str, rng: np.random.Generator
) -> int:

    if L_max < 1:
        raise ValueError("L_max must be at least 1")
    if lag_distribution == "uniform":
        return int(rng.integers(1, L_max + 1))

    draw = 1
    while rng.random() > 0.5 and draw < L_max:
        draw += 1
    if lag_distribution == "short":
        return draw
    if lag_distribution == "long":
        return L_max + 1 - draw
    raise ValueError("lag_distribution must be uniform, short, or long")

def sample_distinct_lags(
    L_max: int,
    lag_distribution: str,
    count: int,
    rng: np.random.Generator,
) -> List[int]:
    """Sample the requested number of distinct lags."""
    if count < 1:
        raise ValueError("count must be at least 1")

    if count > L_max:
        raise ValueError("Cannot draw more distinct lags than L_max")

    selected: List[int] = []

    while len(selected) < count:
        lag = sample_lag(L_max, lag_distribution, rng)

        if lag not in selected:
            selected.append(lag)

    return selected

def sample_random_dag_with_lags(
    n_nodes: int,
    p_edge: float,
    L_max: int,
    rng: np.random.Generator,
    lag_distribution: str,
    multi_lag_fraction: float = 0.0,
    lags_per_multi_edge: int = 2,
) -> Structure:
    """Generate a variable-level DAG and assign one or more lags per adjacency.

    ``p_edge`` controls the number of directed variable-level adjacencies.  When
    ``multi_lag_fraction`` is positive, that fraction of sampled adjacencies is
    selected to carry ``lags_per_multi_edge`` distinct lagged effects.  The
    graph is sampled conditional on containing at least one adjacency so that a
    multi-lag benchmark trial cannot accidentally contain no multi-lag truth.
    """
    if not 0.0 <= p_edge <= 1.0:
        raise ValueError("p_edge must lie in [0, 1]")
    if not 0.0 <= multi_lag_fraction <= 1.0:
        raise ValueError("multi_lag_fraction must lie in [0, 1]")
    if multi_lag_fraction > 0.0 and not 2 <= lags_per_multi_edge <= L_max:
        raise ValueError(
            "For a multi-lag DGP, lags_per_multi_edge must lie in [2, L_max]"
        )

    structure: Structure = {node: [] for node in range(n_nodes)}

    # Retain the original sampling path exactly when multi-lag generation is off.
    if multi_lag_fraction == 0.0:
        for child in range(n_nodes):
            for parent in range(child):
                if rng.random() < p_edge:
                    structure[child].append(
                        (parent, sample_lag(L_max, lag_distribution, rng))
                    )
        return structure

    possible_pairs = [
        (parent, child)
        for child in range(n_nodes)
        for parent in range(child)
    ]
    if not possible_pairs or p_edge == 0.0:
        raise ValueError(
            "A multi-lag setting requires N >= 2 and a positive p_edge"
        )

    adjacency_pairs: List[DirectedPair] = []
    for _ in range(10_000):
        adjacency_pairs = [
            pair for pair in possible_pairs if rng.random() < p_edge
        ]
        if adjacency_pairs:
            break
    if not adjacency_pairs:
        raise RuntimeError("Could not sample a non-empty multi-lag graph")

    n_multi_lag_pairs = max(
        1, int(math.ceil(multi_lag_fraction * len(adjacency_pairs)))
    )
    multi_lag_indices = set(
        int(index)
        for index in rng.choice(
            len(adjacency_pairs), size=n_multi_lag_pairs, replace=False
        )
    )

    for pair_index, (parent, child) in enumerate(adjacency_pairs):
        lag_count = lags_per_multi_edge if pair_index in multi_lag_indices else 1
        for lag in sample_distinct_lags(
            L_max, lag_distribution, lag_count, rng
        ):
            structure[child].append((parent, lag))
    return structure



def sample_coefficients(
    structure: Structure,
    node_types: Mapping[int, str],
    beta_scale: float,
    logit_scale: float,
    rng: np.random.Generator,
) -> Dict[LaggedEdge, float]:
    """Draw an independent coefficient for every lagged causal effect."""
    coefficients: Dict[LaggedEdge, float] = {}
    for child, parents in structure.items():
        scale = logit_scale if node_types[child] == "binary" else beta_scale
        for parent, lag in parents:
            coefficients[(parent, child, lag)] = float(rng.normal(0.0, scale))
    return coefficients


def draw_continuous_noise(
    size: int | Tuple[int, ...],
    standard_deviation: float,
    distribution: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw zero-mean noise with the requested standard deviation."""
    if distribution == "gaussian":
        return rng.normal(0.0, standard_deviation, size=size)
    if distribution == "laplace":
        return rng.laplace(0.0, standard_deviation / math.sqrt(2.0), size=size)
    if distribution == "uniform":
        radius = math.sqrt(3.0) * standard_deviation
        return rng.uniform(-radius, radius, size=size)
    raise ValueError("noise_distribution must be gaussian, laplace, or uniform")


def simulate_time_series(
    config: DGPConfig,
    structure: Structure,
    node_types: Mapping[int, str],
    coefficients: Mapping[LaggedEdge, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Set[UndirectedPair]]:
    n_nodes = config.N
    data = np.zeros((config.T, n_nodes), dtype=float)

    confounder_specs: List[Tuple[int, int, int]] = []
    latent = None
    if config.n_confounders > 0:
        if n_nodes < 2:
            raise ValueError("At least two observed nodes are needed for confounders")
        all_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
        rng.shuffle(all_pairs)
        latent = np.zeros((config.T, config.n_confounders), dtype=float)
        latent_phi = min(max(config.phi / 2.0, 0.0), 0.99)
        for confounder in range(config.n_confounders):
            left, right = all_pairs[confounder % len(all_pairs)]
            confounder_lag = int(rng.integers(1, min(3, config.L_max) + 1))
            confounder_specs.append((left, right, confounder_lag))
            latent[0, confounder] = rng.normal()
            for t in range(1, config.T):
                latent[t, confounder] = (
                    latent_phi * latent[t - 1, confounder] + rng.normal()
                )

    for t in range(config.T):
        for child in range(n_nodes):
            linear_predictor = 0.0
            for parent, lag in structure.get(child, []):
                previous_t = t - lag
                if previous_t >= 0:
                    linear_predictor += (
                        coefficients[(parent, child, lag)]
                        * data[previous_t, parent]
                    )

            if latent is not None:
                for confounder, (left, right, confounder_lag) in enumerate(
                    confounder_specs
                ):
                    previous_t = t - confounder_lag
                    if previous_t >= 0 and child in (left, right):
                        linear_predictor += (
                            config.conf_strength * latent[previous_t, confounder]
                        )

            if node_types[child] == "continuous":
                ar_term = config.phi * data[t - 1, child] if t > 0 else 0.0
                innovation = float(
                    draw_continuous_noise(
                        1,
                        config.noise_std,
                        config.noise_distribution,
                        rng,
                    )[0]
                )
                data[t, child] = ar_term + linear_predictor + innovation
            else:
                probability = float(_sigmoid(np.array([linear_predictor]))[0])
                data[t, child] = rng.binomial(
                    1, np.clip(probability, 1e-6, 1.0 - 1e-6)
                )

    return data, {
        tuple(sorted((left, right))) for left, right, _ in confounder_specs
    }


def apply_missingness_mcar(
    data: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    if rate <= 0.0:
        return data
    output = data.copy()
    output[rng.random(size=output.shape) < rate] = np.nan
    return output


def apply_missingness_mar(
    data: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    if rate <= 0.0:
        return data
    output = data.copy()
    driver = int(rng.integers(0, output.shape[1]))
    driver_values = np.abs(output[:, driver])
    scaled = (driver_values - np.nanmin(driver_values)) / (
        np.nanmax(driver_values) - np.nanmin(driver_values) + 1e-9
    )
    base_rate = rate / 2.0
    row_probability = np.clip(base_rate + scaled * (rate - base_rate), 0.0, 0.9)
    output[rng.random(size=output.shape) < row_probability[:, None]] = np.nan
    return output


def simple_impute(data: np.ndarray, node_types: Mapping[int, str]) -> np.ndarray:
    output = data.copy()
    for node in range(output.shape[1]):
        column = output[:, node]
        missing = np.isnan(column)
        if not np.any(missing):
            continue
        observed = column[~missing]
        if observed.size == 0:
            fill = 0.0
        elif node_types[node] == "binary":
            fill = float(np.mean(observed) >= 0.5)
        else:
            fill = float(np.mean(observed))
        column[missing] = fill
        output[:, node] = column
    return output


def structure_to_lagged_edges(structure: Structure) -> Set[LaggedEdge]:
    return {
        (int(parent), int(child), int(lag))
        for child, parents in structure.items()
        for parent, lag in parents
    }


def simulate_trial(
    config: DGPConfig,
    lag_distribution: str,
    seed: int,
) -> SimulatedTrial:
    rng = np.random.default_rng(seed)
    node_types = sample_node_types(config.N, config.frac_binary, rng)
    structure = sample_random_dag_with_lags(
        config.N,
        config.p_edge,
        config.L_max,
        rng,
        lag_distribution,
        multi_lag_fraction=config.multi_lag_fraction,
        lags_per_multi_edge=config.lags_per_multi_edge,
    )
    coefficients = sample_coefficients(
        structure,
        node_types,
        config.beta_scale,
        config.logit_scale,
        rng,
    )
    data, confounded_pairs = simulate_time_series(
        config, structure, node_types, coefficients, rng
    )
    data = apply_missingness_mcar(data, config.mcar_rate, rng)
    data = apply_missingness_mar(data, config.mar_rate, rng)
    if config.mcar_rate > 0.0 or config.mar_rate > 0.0:
        data = simple_impute(data, node_types)

    true_edges = structure_to_lagged_edges(structure)
    if config.phi != 0.0:
        for node, node_type in node_types.items():
            if node_type == "continuous":
                true_edges.add((node, node, 1))

    return SimulatedTrial(
        data=data,
        node_types=node_types,
        true_structure=structure,
        true_edges=true_edges,
        confounded_pairs=confounded_pairs,
    )


# ---------------------------------------------------------------------------
# Same-score fixed-common-lag baseline
# ---------------------------------------------------------------------------


class FixedLagLocalScorer:
    
    def __init__(
        self,
        data: np.ndarray,
        node_types: Mapping[int, str],
        fixed_lag: int,
        penalty_coef: float,
    ) -> None:
        self.data = np.asarray(data, dtype=float)
        self.node_types = dict(node_types)
        self.fixed_lag = int(fixed_lag)
        self.penalty_coef = float(penalty_coef)
        self.T, self.N = self.data.shape
        self.cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}

    def local_score(self, child: int, parents: Iterable[int]) -> float:
        parent_tuple = tuple(sorted(int(parent) for parent in parents))
        key = (int(child), parent_tuple)
        if key in self.cache:
            return self.cache[key]

        lag = self.fixed_lag if parent_tuple else 0
        n_effective = self.T - lag
        if n_effective <= 1:
            return -1e12
        y_raw = self.data[lag:, child]

        columns: List[np.ndarray] = []
        for parent in parent_tuple:
            parent_series = self.data[: self.T - self.fixed_lag, parent]
            if self.node_types[parent] == "binary":
                parent_series = _binary01(parent_series, f"parent {parent}")
            columns.append(parent_series.reshape(-1, 1))

        if columns:
            design = np.hstack(columns)
            design = np.hstack([np.ones((n_effective, 1)), design])
        else:
            design = np.ones((n_effective, 1))

        if self.node_types[child] == "continuous":
            beta = np.linalg.lstsq(design, y_raw, rcond=None)[0]
            residuals = y_raw - design @ beta
            sigma2 = max(float(np.sum(residuals**2)) / n_effective, 1e-9)
            loglik = -0.5 * n_effective * (
                math.log(2.0 * math.pi * sigma2) + 1.0
            )
        else:
            y = _binary01(y_raw, f"child {child}")
            _, loglik = _logistic_irls(design, y)

        n_parameters = design.shape[1]
        bic = 2.0 * loglik - n_parameters * math.log(max(n_effective, 2))
        lag_penalty = len(parent_tuple) * max(0, self.fixed_lag - 1)
        value = float(bic - self.penalty_coef * lag_penalty)
        self.cache[key] = value
        return value


def fixed_lag_tabu_search(
    data: np.ndarray,
    node_types: Mapping[int, str],
    fixed_lag: int,
    penalty_coef: float,
    max_iter: int,
    tabu_length: int,
) -> Tuple[Structure, float]:
    """Search adjacencies while constraining every selected edge to a single fixed lag."""
    n_nodes = data.shape[1]
    scorer = FixedLagLocalScorer(data, node_types, fixed_lag, penalty_coef)
    parents: Dict[int, Set[int]] = {node: set() for node in range(n_nodes)}
    local_scores = np.array(
        [scorer.local_score(node, ()) for node in range(n_nodes)], dtype=float
    )
    current_score = float(local_scores.sum())
    best_parents = {node: set(values) for node, values in parents.items()}
    best_score = current_score
    tabu: List[Tuple[str, int, int]] = []

    def candidate(
        action: str, parent: int, child: int
    ) -> Tuple[float, Dict[int, Set[int]], Dict[int, float]]:
        changes: Dict[int, Set[int]] = {}
        updated_scores: Dict[int, float] = {}
        if action == "add":
            new_child_parents = set(parents[child])
            new_child_parents.add(parent)
            changes[child] = new_child_parents
        elif action == "remove":
            new_child_parents = set(parents[child])
            new_child_parents.remove(parent)
            changes[child] = new_child_parents
        elif action == "reverse":
            new_child_parents = set(parents[child])
            new_child_parents.remove(parent)
            new_parent_parents = set(parents[parent])
            new_parent_parents.add(child)
            changes[child] = new_child_parents
            changes[parent] = new_parent_parents
        else:
            raise ValueError(f"Unknown move {action}")

        score_value = current_score
        for changed_child, changed_parents in changes.items():
            new_local = scorer.local_score(changed_child, changed_parents)
            updated_scores[changed_child] = new_local
            score_value += new_local - local_scores[changed_child]
        return float(score_value), changes, updated_scores

    def all_moves() -> Iterable[Tuple[str, int, int]]:
        for parent in range(n_nodes):
            for child in range(n_nodes):
                if parent not in parents[child]:
                    yield ("add", parent, child)
                else:
                    yield ("remove", parent, child)
                    if (
                        parent != child
                        and child not in parents[parent]
                    ):
                        yield ("reverse", parent, child)

    def apply_move(
        changes: Mapping[int, Set[int]], updated_scores: Mapping[int, float]
    ) -> None:
        nonlocal current_score
        for changed_child, changed_parents in changes.items():
            parents[changed_child] = set(changed_parents)
            local_scores[changed_child] = updated_scores[changed_child]
        current_score = float(local_scores.sum())

    # Greedy initialisation.
    while True:
        best_candidate = None
        best_candidate_score = current_score
        for move in all_moves():
            value, changes, updated = candidate(*move)
            if value > best_candidate_score:
                best_candidate_score = value
                best_candidate = (move, changes, updated)
        if best_candidate is None:
            break
        _, changes, updated = best_candidate
        apply_move(changes, updated)
        if current_score > best_score:
            best_score = current_score
            best_parents = {node: set(values) for node, values in parents.items()}

    # Tabu phase.
    for _ in range(max(0, int(max_iter))):
        selected = None
        selected_score = -float("inf")
        for move in all_moves():
            value, changes, updated = candidate(*move)
            admissible = move not in tabu or value > best_score
            if admissible and value > selected_score:
                selected_score = value
                selected = (move, changes, updated)
        if selected is None:
            break
        move, changes, updated = selected
        apply_move(changes, updated)

        action, parent, child = move
        if action == "add":
            inverse = ("remove", parent, child)
        elif action == "remove":
            inverse = ("add", parent, child)
        else:
            inverse = ("reverse", child, parent)
        tabu.append(inverse)
        if len(tabu) > max(1, int(tabu_length)):
            tabu.pop(0)

        if current_score > best_score:
            best_score = current_score
            best_parents = {node: set(values) for node, values in parents.items()}

    structure: Structure = {
        child: [(parent, fixed_lag) for parent in sorted(parent_set)]
        for child, parent_set in best_parents.items()
    }
    return structure, float(best_score)


def validation_predictive_loglik(
    data: np.ndarray,
    split: int,
    structure: Structure,
    node_types: Mapping[int, str],
) -> float:
    """Evaluates how well a graph predicts held-out observations."""
    T, n_nodes = data.shape
    total = 0.0
    n_terms = 0
    for child in range(n_nodes):
        parents = structure.get(child, [])
        maximum_lag = max((lag for _, lag in parents), default=0)
        train_times = np.arange(maximum_lag, split)
        validation_times = np.arange(split, T)
        if train_times.size <= len(parents) + 2 or validation_times.size == 0:
            return -float("inf")

        def design_for(times: np.ndarray) -> np.ndarray:
            columns = [np.ones(times.size)]
            for parent, lag in parents:
                values = data[times - lag, parent]
                if node_types[parent] == "binary":
                    values = _binary01(values, f"parent {parent}")
                columns.append(values.astype(float))
            return np.column_stack(columns)

        X_train = design_for(train_times)
        X_validation = design_for(validation_times)
        y_train_raw = data[train_times, child]
        y_validation_raw = data[validation_times, child]

        if node_types[child] == "continuous":
            beta = np.linalg.lstsq(X_train, y_train_raw, rcond=None)[0]
            train_residuals = y_train_raw - X_train @ beta
            sigma2 = max(float(np.mean(train_residuals**2)), 1e-9)
            validation_residuals = y_validation_raw - X_validation @ beta
            total += float(
                np.sum(
                    -0.5
                    * (
                        np.log(2.0 * np.pi * sigma2)
                        + (validation_residuals**2) / sigma2
                    )
                )
            )
        else:
            y_train = _binary01(y_train_raw, f"child {child}")
            y_validation = _binary01(y_validation_raw, f"child {child}")
            beta, _ = _logistic_irls(X_train, y_train)
            eta = X_validation @ beta
            total += float(np.sum(y_validation * eta - _log1pexp(eta)))
        n_terms += validation_times.size
    return total / max(n_terms, 1)


def run_fixed_lag_cv(
    data: np.ndarray,
    node_types: Mapping[int, str],
    L_max: int,
    penalty_coef: float,
    max_iter: int,
    tabu_length: int,
    validation_fraction: float,
) -> MethodOutput:
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must lie between 0.05 and 0.5")
    split = int(math.floor(data.shape[0] * (1.0 - validation_fraction)))
    if split <= L_max + 5 or data.shape[0] - split < 5:
        raise ValueError("Time series is too short for fixed-lag validation")

    candidate_scores: Dict[int, float] = {}
    training = data[:split]
    for lag in range(1, L_max + 1):
        structure, _ = fixed_lag_tabu_search(
            training,
            node_types,
            fixed_lag=lag,
            penalty_coef=penalty_coef,
            max_iter=max_iter,
            tabu_length=tabu_length,
        )
        candidate_scores[lag] = validation_predictive_loglik(
            data, split, structure, node_types
        )

    selected_lag = max(candidate_scores, key=candidate_scores.get)
    final_structure, final_score = fixed_lag_tabu_search(
        data,
        node_types,
        fixed_lag=selected_lag,
        penalty_coef=penalty_coef,
        max_iter=max_iter,
        tabu_length=tabu_length,
    )
    return MethodOutput(
        lagged_edges=structure_to_lagged_edges(final_structure),
        contemporaneous_edges=set(),
        score=final_score,
        metadata={
            "selected_lag": selected_lag,
            "validation_split": split,
            "validation_loglik_by_lag": candidate_scores,
        },
    )


# ---------------------------------------------------------------------------
# Load variable lag tabu
# ---------------------------------------------------------------------------


def resolve_local_file(explicit: Optional[str], candidates: Sequence[str]) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    script_dir = Path(__file__).resolve().parent
    roots = (script_dir, script_dir / "upload", Path.cwd(), Path.cwd() / "upload")
    for root in roots:
        for candidate in candidates:
            path = (root / candidate).resolve()
            if path.is_file():
                return path
    joined = ", ".join(candidates)
    raise FileNotFoundError(
        f"Could not locate any of [{joined}]. Pass the path explicitly."
    )


def load_python_file(path: Path, module_name: str) -> Any:
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


class ProposedLearner:
    def __init__(self, args: argparse.Namespace) -> None:
        base_path = resolve_local_file(
            args.learner_file,
            ("variable_lag_tabu.py",),
        )
        self.base_path = base_path
        self.base_module = load_python_file(base_path, "variable_lag_tabu")
        self.parallel_module = None
        if args.proposed_parallel:
            parallel_path = resolve_local_file(
                args.parallel_learner_file,
                ("variable_lag_tabu_parallel.py",),
            )
            self.parallel_module = load_python_file(
                parallel_path, "variable_lag_tabu_parallel"
            )
            self.parallel_path = parallel_path

    def run(
        self,
        data: np.ndarray,
        node_types: Mapping[int, str],
        seed: int,
        L_max: int,
        args: argparse.Namespace,
    ) -> MethodOutput:
        if self.parallel_module is not None:
            structure, score = self.parallel_module.tabu_search_variable_lag_parallel(
                data,
                dict(node_types),
                L_max=L_max,
                penalty_coef=args.penalty_coef,
                tabu_max_iter=args.max_iter,
                tabu_length=args.tabu_length,
                n_jobs=args.proposed_jobs,
            )
            implementation = str(self.parallel_path)
        else:
            structure, score = self.base_module.tabu_search_greedy_lag(
                data,
                dict(node_types),
                max_iter=args.max_iter,
                tabu_length=args.tabu_length,
                penalty_coef=args.penalty_coef,
                L_max=L_max,
                seed=seed,
            )
            implementation = str(self.base_path)
        return MethodOutput(
            lagged_edges=structure_to_lagged_edges(structure),
            contemporaneous_edges=set(),
            score=float(score),
            metadata={"implementation": implementation},
        )


# ---------------------------------------------------------------------------
# External method adapters
# ---------------------------------------------------------------------------


def graph_marks_to_edges(graph: np.ndarray, L_max: int) -> Set[LaggedEdge]:
    graph = np.asarray(graph)
    if graph.ndim != 3:
        raise ValueError(f"Expected a 3D Tigramite graph array, got {graph.shape}")
    n_nodes = graph.shape[0]
    edges: Set[LaggedEdge] = set()
    maximum_lag = min(L_max, graph.shape[2] - 1)
    for parent in range(n_nodes):
        for child in range(n_nodes):
            for lag in range(1, maximum_lag + 1):
                if str(graph[parent, child, lag]).strip():
                    edges.add((parent, child, lag))
    return edges


def graph_marks_to_contemporaneous(graph: np.ndarray) -> Set[DirectedPair]:
    graph = np.asarray(graph)
    output: Set[DirectedPair] = set()
    if graph.ndim != 3 or graph.shape[2] == 0:
        return output
    for source in range(graph.shape[0]):
        for target in range(graph.shape[1]):
            if source != target and str(graph[source, target, 0]).strip():
                output.add((source, target))
    return output


def run_pcmci(data: np.ndarray, config: DGPConfig, args: argparse.Namespace) -> MethodOutput:
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise SkipMethod("PCMCI requires the optional package 'tigramite'") from exc

    dataframe = pp.DataFrame(np.asarray(data, dtype=float))
    independence_test = ParCorr(significance="analytic")
    learner = PCMCI(
        dataframe=dataframe,
        cond_ind_test=independence_test,
        verbosity=args.external_verbosity,
    )
    results = learner.run_pcmci(
        tau_min=1,
        tau_max=config.L_max,
        pc_alpha=args.pcmci_pc_alpha,
        alpha_level=args.pcmci_alpha,
        fdr_method=args.pcmci_fdr,
    )
    graph = results.get("graph")
    if graph is None:
        p_matrix = np.asarray(results["p_matrix"])
        graph = np.where(p_matrix <= args.pcmci_alpha, "-->", "")
    return MethodOutput(
        lagged_edges=graph_marks_to_edges(graph, config.L_max),
        contemporaneous_edges=set(),
        metadata={
            "pc_alpha": args.pcmci_pc_alpha,
            "alpha_level": args.pcmci_alpha,
            "fdr_method": args.pcmci_fdr,
            "ci_test": "ParCorr",
        },
    )


def run_lpcmci(data: np.ndarray, config: DGPConfig, args: argparse.Namespace) -> MethodOutput:
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.lpcmci import LPCMCI
    except ImportError as exc:
        raise SkipMethod("LPCMCI requires the optional package 'tigramite'") from exc

    dataframe = pp.DataFrame(np.asarray(data, dtype=float))
    independence_test = ParCorr(significance="analytic")
    learner = LPCMCI(
        dataframe=dataframe,
        cond_ind_test=independence_test,
        verbosity=args.external_verbosity,
    )
    tau_min = 0 if args.lpcmci_include_contemporaneous else 1
    max_p_global = (
    np.inf
    if args.lpcmci_max_p_global is None
        else args.lpcmci_max_p_global
    )

    max_p_non_ancestral = (
        np.inf
        if args.lpcmci_max_p_non_ancestral is None
            else args.lpcmci_max_p_non_ancestral
    )

    max_q_global = (
        np.inf
        if args.lpcmci_max_q_global is None
            else args.lpcmci_max_q_global
    )

    results = learner.run_lpcmci(
        tau_min=tau_min,
        tau_max=config.L_max,
        pc_alpha=args.lpcmci_alpha,
        max_p_global=max_p_global,
        max_p_non_ancestral=max_p_non_ancestral,
        max_q_global=max_q_global,
    )
    graph = np.asarray(results["graph"])
    mark_counts: Dict[str, int] = {}
    for mark in graph.ravel():
        text = str(mark).strip()
        if text:
            mark_counts[text] = mark_counts.get(text, 0) + 1
    return MethodOutput(
        lagged_edges=graph_marks_to_edges(graph, config.L_max),
        contemporaneous_edges=graph_marks_to_contemporaneous(graph),
        metadata={
            "pc_alpha": args.lpcmci_alpha,
            "tau_min": tau_min,
            "max_p_global": max_p_global,
            "max_p_non_ancestral": max_p_non_ancestral,
            "max_q_global": max_q_global,
            "ci_test": "ParCorr",
            "pag_mark_counts": mark_counts,
        },
    )


_DYNAMIC_NODE = re.compile(r"^(?P<variable>.+)_lag(?P<lag>[0-9]+)$")


def run_dynotears(
    data: np.ndarray, config: DGPConfig, args: argparse.Namespace
) -> MethodOutput:
    try:
        from causalnex.structure.dynotears import from_pandas_dynamic
    except ImportError as exc:
        raise SkipMethod(
            "DYNOTEARS requires 'causalnex' (use its supported Python 3.10 environment)"
        ) from exc

    columns = [f"X{node}" for node in range(config.N)]
    frame = pd.DataFrame(np.asarray(data, dtype=float), columns=columns)
    tabu_edges = [
        (0, source, target)
        for source in columns
        for target in columns
        if source != target
    ]
    graph = from_pandas_dynamic(
        frame,
        p=config.L_max,
        lambda_w=args.dynotears_lambda_w,
        lambda_a=args.dynotears_lambda_a,
        max_iter=args.dynotears_max_iter,
        h_tol=args.dynotears_h_tol,
        w_threshold=args.dynotears_weight_threshold,
        tabu_edges=tabu_edges,
    )

    lagged: Set[LaggedEdge] = set()
    contemporaneous: Set[DirectedPair] = set()
    weights: Dict[str, float] = {}
    column_index = {name: index for index, name in enumerate(columns)}
    for source, target, attributes in graph.edges(data=True):
        source_match = _DYNAMIC_NODE.match(str(source))
        target_match = _DYNAMIC_NODE.match(str(target))
        if source_match is None or target_match is None:
            raise ValueError(f"Unexpected DYNOTEARS node labels: {source!r}, {target!r}")
        source_name = source_match.group("variable")
        target_name = target_match.group("variable")
        source_lag = int(source_match.group("lag"))
        target_lag = int(target_match.group("lag"))
        if source_name not in column_index or target_name not in column_index:
            raise ValueError("DYNOTEARS returned an unknown variable label")
        parent = column_index[source_name]
        child = column_index[target_name]
        weight = float(attributes.get("weight", float("nan")))
        if target_lag != 0:
            continue
        if source_lag == 0:
            contemporaneous.add((parent, child))
        elif source_lag <= config.L_max:
            lagged.add((parent, child, source_lag))
            weights[f"{parent}->{child}@{source_lag}"] = weight

    return MethodOutput(
        lagged_edges=lagged,
        contemporaneous_edges=contemporaneous,
        metadata={
            "lambda_w": args.dynotears_lambda_w,
            "lambda_a": args.dynotears_lambda_a,
            "weight_threshold": args.dynotears_weight_threshold,
            "contemporaneous_edges_forbidden": True,
            "edge_weights": weights,
        },
    )


def run_varlingam(
    data: np.ndarray,
    config: DGPConfig,
    seed: int,
    args: argparse.Namespace,
) -> MethodOutput:
    try:
        from lingam import VARLiNGAM
    except ImportError as exc:
        raise SkipMethod("VARLiNGAM requires the optional package 'lingam'") from exc

    learner = VARLiNGAM(
        lags=config.L_max,
        criterion=None,
        prune=args.varlingam_prune,
        random_state=seed,
    )
    learner.fit(np.asarray(data, dtype=float))
    matrices = np.asarray(learner.adjacency_matrices_, dtype=float)
    if matrices.ndim != 3:
        raise ValueError(f"Unexpected VARLiNGAM matrix shape {matrices.shape}")

    lagged: Set[LaggedEdge] = set()
    contemporaneous: Set[DirectedPair] = set()
    for matrix_index, matrix in enumerate(matrices):
        for child in range(config.N):
            for parent in range(config.N):
                if abs(float(matrix[child, parent])) < args.varlingam_threshold:
                    continue
                if matrix_index == 0:
                    if parent != child:
                        contemporaneous.add((parent, child))
                elif matrix_index <= config.L_max:
                    lagged.add((parent, child, matrix_index))
    return MethodOutput(
        lagged_edges=lagged,
        contemporaneous_edges=contemporaneous,
        metadata={
            "threshold": args.varlingam_threshold,
            "prune": args.varlingam_prune,
            "criterion": None,
        },
    )

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def collapse_adjacencies(edges: Iterable[LaggedEdge]) -> Set[DirectedPair]:
    return {(parent, child) for parent, child, lag in edges if lag >= 1}


def precision_recall_f1(
    truth: Set[Any], prediction: Set[Any]
) -> Tuple[float, float, float, int, int, int]:
    true_positives = len(truth & prediction)
    false_positives = len(prediction - truth)
    false_negatives = len(truth - prediction)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, true_positives, false_positives, false_negatives


def directed_adjacency_shd(
    truth: Set[DirectedPair], prediction: Set[DirectedPair]
) -> int:
    nodes = {node for edge in truth | prediction for node in edge}
    cost = 0


    for node in nodes:
        cost += int(((node, node) in truth) != ((node, node) in prediction))

    for left in sorted(nodes):
        for right in sorted(node for node in nodes if node > left):
            truth_directions = {
                edge
                for edge in ((left, right), (right, left))
                if edge in truth
            }
            predicted_directions = {
                edge
                for edge in ((left, right), (right, left))
                if edge in prediction
            }
            if (
                len(truth_directions) == 1
                and len(predicted_directions) == 1
                and truth_directions != predicted_directions
            ):
                cost += 1
            else:
                cost += len(truth_directions.symmetric_difference(predicted_directions))
    return cost


def balanced_scoring_function(
    truth: Set[DirectedPair], prediction: Set[DirectedPair], n_nodes: int
) -> float:
    universe = {(parent, child) for parent in range(n_nodes) for child in range(n_nodes)}
    nonedges = universe - truth
    if not truth or not nonedges:
        return float("nan")
    true_positives = len(truth & prediction)
    false_positives = len(prediction & nonedges)
    false_negatives = len(truth - prediction)
    true_negatives = len(nonedges - prediction)
    return 0.5 * (
        true_positives / len(truth)
        + true_negatives / len(nonedges)
        - false_positives / len(nonedges)
        - false_negatives / len(truth)
    )


def conditional_lag_mae(
    truth: Set[LaggedEdge], prediction: Set[LaggedEdge]
) -> Tuple[float, int]:
    true_by_pair: Dict[DirectedPair, Set[int]] = {}
    predicted_by_pair: Dict[DirectedPair, Set[int]] = {}
    for parent, child, lag in truth:
        true_by_pair.setdefault((parent, child), set()).add(lag)
    for parent, child, lag in prediction:
        predicted_by_pair.setdefault((parent, child), set()).add(lag)
    common_pairs = set(true_by_pair) & set(predicted_by_pair)
    if not common_pairs:
        return float("nan"), 0
    errors = []
    for pair in common_pairs:
        errors.append(
            min(
                abs(true_lag - predicted_lag)
                for true_lag in true_by_pair[pair]
                for predicted_lag in predicted_by_pair[pair]
            )
        )
    return float(np.mean(errors)), len(common_pairs)

def multi_lag_recovery_metrics(
    truth: Set[LaggedEdge],
    prediction: Set[LaggedEdge],
) -> Dict[str, Any]:
    true_by_pair: Dict[DirectedPair, Set[int]] = {}
    predicted_by_pair: Dict[DirectedPair, Set[int]] = {}

    for parent, child, lag in truth:
        true_by_pair.setdefault(
            (parent, child), set()
        ).add(lag)

    for parent, child, lag in prediction:
        predicted_by_pair.setdefault(
            (parent, child), set()
        ).add(lag)

    multi_lag_truth = {
        pair: lags
        for pair, lags in true_by_pair.items()
        if len(lags) > 1
    }

    n_pairs = len(multi_lag_truth)
    n_edges = sum(
        len(lags)
        for lags in multi_lag_truth.values()
    )

    if n_pairs == 0:
        return {
            "multi_lag_edge_recall": float("nan"),
            "multi_lag_pair_any_recall": float("nan"),
            "multi_lag_pair_exact_set_accuracy": float("nan"),
            "n_true_multi_lag_adjacencies": 0,
            "n_true_edges_on_multi_lag_adjacencies": 0,
        }

    recovered_edges = 0
    pairs_with_any = 0
    pairs_with_exact_set = 0

    for pair, true_lags in multi_lag_truth.items():
        predicted_lags = predicted_by_pair.get(pair, set())

        recovered_edges += len(
            true_lags & predicted_lags
        )

        pairs_with_any += int(
            bool(true_lags & predicted_lags)
        )

        pairs_with_exact_set += int(
            predicted_lags == true_lags
        )

    return {
        "multi_lag_edge_recall":
            recovered_edges / n_edges,
        "multi_lag_pair_any_recall":
            pairs_with_any / n_pairs,
        "multi_lag_pair_exact_set_accuracy":
            pairs_with_exact_set / n_pairs,
        "n_true_multi_lag_adjacencies":
            n_pairs,
        "n_true_edges_on_multi_lag_adjacencies":
            n_edges,
    }

def normalize_undirected(edges: Iterable[DirectedPair]) -> Set[UndirectedPair]:
    return {tuple(sorted((left, right))) for left, right in edges if left != right}


def evaluate_method(
    truth: SimulatedTrial,
    prediction: MethodOutput,
    runtime_seconds: float,
    config: DGPConfig,
) -> Dict[str, Any]:
    true_exact = set(truth.true_edges)
    predicted_exact = {
        edge for edge in prediction.lagged_edges if 1 <= edge[2] <= config.L_max
    }
    true_adjacency = collapse_adjacencies(true_exact)
    predicted_adjacency = collapse_adjacencies(predicted_exact)

    ap, ar, af1, atp, afp, afn = precision_recall_f1(
        true_adjacency, predicted_adjacency
    )
    ep, er, ef1, etp, efp, efn = precision_recall_f1(true_exact, predicted_exact)
    lag_mae, matched_pairs = conditional_lag_mae(true_exact, predicted_exact)
    multi_lag_metrics = multi_lag_recovery_metrics(true_exact, predicted_exact)
    true_contemporary = set(truth.confounded_pairs)
    predicted_contemporary = normalize_undirected(prediction.contemporaneous_edges)
    if true_contemporary or predicted_contemporary:
        cp, cr, cf1, ctp, cfp, cfn = precision_recall_f1(
            true_contemporary, predicted_contemporary
        )
    else:
        cp = cr = cf1 = float("nan")
        ctp = cfp = cfn = 0

    return {
        "adjacency_precision": ap,
        "adjacency_recall": ar,
        "adjacency_f1": af1,
        "adjacency_tp": atp,
        "adjacency_fp": afp,
        "adjacency_fn": afn,
        "adjacency_shd": directed_adjacency_shd(
            true_adjacency, predicted_adjacency
        ),
        "adjacency_bsf": balanced_scoring_function(
            true_adjacency, predicted_adjacency, config.N
        ),
        "exact_precision": ep,
        "exact_recall": er,
        "exact_f1": ef1,
        "exact_tp": etp,
        "exact_fp": efp,
        "exact_fn": efn,
        "exact_shd": len(true_exact.symmetric_difference(predicted_exact)),
        "lag_mae_conditional": lag_mae,
        "lag_mae_matched_pairs": matched_pairs,
        **multi_lag_metrics,
        "contemp_skeleton_precision": cp,
        "contemp_skeleton_recall": cr,
        "contemp_skeleton_f1": cf1,
        "contemp_skeleton_tp": ctp,
        "contemp_skeleton_fp": cfp,
        "contemp_skeleton_fn": cfn,
        "n_true_adjacencies": len(true_adjacency),
        "n_predicted_adjacencies": len(predicted_adjacency),
        "n_true_lagged_edges": len(true_exact),
        "n_predicted_lagged_edges": len(predicted_exact),
        "n_predicted_contemporaneous_edges": len(prediction.contemporaneous_edges),
        "runtime_seconds": runtime_seconds,
        "learner_score": prediction.score,
        "selected_fixed_lag": prediction.metadata.get("selected_lag", float("nan")),
        "method_metadata": json.dumps(prediction.metadata, sort_keys=True, default=str),
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def method_eligibility(
    method: str, config: DGPConfig, args: argparse.Namespace
) -> Optional[str]:
    continuous_only = {"pcmci", "dynotears", "lpcmci", "varlingam"}
    if method in continuous_only and config.frac_binary > 0.0:
        return "method is evaluated only on all-continuous simulations"

    if config.n_confounders > 0 and method in {
        "pcmci",
        "dynotears",
        "varlingam",
    }:
        if not args.allow_assumption_violations:
            return "latent-confounded setting is reserved for LPCMCI"

    if method == "lpcmci" and config.n_confounders == 0:
        if not args.lpcmci_all_continuous:
            return "LPCMCI is reserved for the latent-confounded continuous setting"

    if method == "varlingam" and config.noise_distribution == "gaussian":
        if not args.allow_assumption_violations:
            return "Gaussian innovations do not satisfy VARLiNGAM's non-Gaussian assumption"
    return None


def package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_report(args: argparse.Namespace) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "tigramite": package_version("tigramite"),
        "causalnex": package_version("causalnex"),
        "lingam": package_version("lingam"),
    }
    try:
        report["learner_file"] = str(
            resolve_local_file(
                args.learner_file,
                ("variable_lag_tabu.py",),
            )
        )
    except Exception as exc:
        report["learner_file"] = None
        report["learner_error"] = str(exc)
    return report


def parse_methods(raw_methods: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for item in raw_methods:
        expanded.extend(part.strip() for part in item.split(",") if part.strip())
    if "all" in expanded:
        expanded = list(ALL_METHODS)
    unknown = sorted(set(expanded) - set(ALL_METHODS))
    if unknown:
        raise ValueError("Unknown methods: " + ", ".join(unknown))
    output = []
    for method in expanded:
        if method not in output:
            output.append(method)
    return output


def base_config_from_args(args: argparse.Namespace) -> DGPConfig:
    return DGPConfig(
        N=args.N,
        T=args.T,
        L_max=args.L_max,
        p_edge=args.p_edge,
        frac_binary=args.frac_binary,
        beta_scale=args.beta_scale,
        logit_scale=args.logit_scale,
        noise_std=args.noise_std,
        noise_distribution=args.noise_distribution,
        phi=args.phi,
        n_confounders=args.n_confounders,
        conf_strength=args.conf_strength,
        mcar_rate=args.mcar_rate,
        mar_rate=args.mar_rate,
        multi_lag_fraction=(
            0.0
            if args.multi_lag_fraction is None
            else args.multi_lag_fraction
        ),
        lags_per_multi_edge=args.lags_per_multi_edge,
    )


def apply_sweep_value(config: DGPConfig, sweep: str, value: str) -> DGPConfig:
    if sweep == "T":
        return replace(config, T=int(value))
    if sweep == "N":
        return replace(config, N=int(value))
    if sweep == "density":
        return replace(config, p_edge=float(value))
    if sweep == "noise":
        return replace(config, noise_std=float(value))
    if sweep == "phi":
        return replace(config, phi=float(value))
    if sweep == "fracbin":
        return replace(config, frac_binary=float(value))
    if sweep == "conf":
        return replace(config, n_confounders=int(value))
    if sweep == "mcar":
        return replace(config, mcar_rate=float(value))
    if sweep == "mar":
        return replace(config, mar_rate=float(value))
    if sweep == "noise_distribution":
        return replace(config, noise_distribution=value)
    if sweep == "lagdist":
        return config
    if sweep == "none":
        return config
    raise ValueError(f"Unknown sweep {sweep}")


def build_settings(
    args: argparse.Namespace,
) -> List[Tuple[str, DGPConfig, str]]:
    base = base_config_from_args(args)
    multi_lag_fraction = (
        0.5 if args.multi_lag_fraction is None else args.multi_lag_fraction
    )
    continuous_multilag = replace(
        base,
        frac_binary=0.0,
        n_confounders=0,
        multi_lag_fraction=multi_lag_fraction,
    )

    if args.suite == "multilag":
        return [
            ("continuous_multilag", continuous_multilag, args.lag_distribution)
        ]

    if args.suite == "reviewer":
        continuous = replace(
            base,
            frac_binary=0.0,
            n_confounders=0,
            multi_lag_fraction=0.0,
        )
        mixed = replace(base, n_confounders=0, multi_lag_fraction=0.0)
        confounder_count = max(1, base.n_confounders or 2)
        return [
            ("mixed_uniform", mixed, "uniform"),
            ("mixed_short", mixed, "short"),
            ("mixed_long", mixed, "long"),
            ("continuous_uniform", continuous, "uniform"),
            ("continuous_short", continuous, "short"),
            ("continuous_long", continuous, "long"),
            (
                "continuous_confounded",
                replace(continuous, n_confounders=confounder_count),
                "uniform",
            ),
            (
                "continuous_laplace",
                replace(continuous, noise_distribution="laplace"),
                "uniform",
            ),
            ("continuous_multilag", continuous_multilag, "uniform"),
        ]

    if args.suite == "quick":
        quick_base = replace(
            base,
            N=min(base.N, 4),
            T=min(base.T, 150),
            L_max=min(base.L_max, 3),
            n_confounders=0,
        )
        return [
            ("quick_mixed", quick_base, "uniform"),
            ("quick_continuous", replace(quick_base, frac_binary=0.0), "uniform"),
        ]

    if args.sweep == "none":
        return [("custom", base, args.lag_distribution)]
    if not args.values:
        raise ValueError("--values is required when --sweep is not 'none'")
    settings = []
    for value in args.values:
        config = apply_sweep_value(base, args.sweep, value)
        lag_distribution = value if args.sweep == "lagdist" else args.lag_distribution
        settings.append((f"{args.sweep}={value}", config, lag_distribution))
    return settings


def bootstrap_mean_ci(
    values: Sequence[float], seed: int, repetitions: int = 5000
) -> Tuple[float, float, float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1)) if n > 1 else 0.0
    if n == 1:
        return mean, mean, mean, standard_deviation, n
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(repetitions, n))
    bootstrap_means = array[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(lower), float(upper), standard_deviation, n


def make_summary(trials: pd.DataFrame, seed: int) -> pd.DataFrame:
    successful = trials.loc[trials["status"] == "ok"].copy()
    rows: List[Dict[str, Any]] = []
    if successful.empty:
        columns = ["setting", "method", "n_success"]
        for metric in PRIMARY_METRICS:
            columns.extend(
                [
                    f"{metric}_mean",
                    f"{metric}_ci_low",
                    f"{metric}_ci_high",
                    f"{metric}_sd",
                    f"{metric}_n",
                ]
            )
        return pd.DataFrame(columns=columns)
    for group_index, ((setting, method), group) in enumerate(
        successful.groupby(["setting", "method"], sort=False)
    ):
        row: Dict[str, Any] = {
            "setting": setting,
            "method": method,
            "n_success": len(group),
        }
        for metric in PRIMARY_METRICS:
            mean, lower, upper, standard_deviation, n_valid = bootstrap_mean_ci(
                group[metric].to_numpy(dtype=float),
                seed=seed + group_index * 1009 + len(metric),
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lower
            row[f"{metric}_ci_high"] = upper
            row[f"{metric}_sd"] = standard_deviation
            row[f"{metric}_n"] = n_valid
        rows.append(row)
    return pd.DataFrame(rows)


def make_paired_differences(trials: pd.DataFrame, seed: int) -> pd.DataFrame:
    columns = [
        "setting",
        "baseline",
        "metric",
        "contrast",
        "mean_difference",
        "ci_low",
        "ci_high",
        "sd",
        "n_pairs",
        "positive_favours_proposed",
    ]
    successful = trials.loc[trials["status"] == "ok"].copy()
    proposed = successful.loc[successful["method"] == "proposed"]
    if proposed.empty:
        return pd.DataFrame(columns=columns)
    metrics = (
        "adjacency_f1",
        "exact_f1",
        "adjacency_shd",
        "exact_shd",
        "lag_mae_conditional",
        "runtime_seconds",
    )
    rows: List[Dict[str, Any]] = []
    key = ["setting", "trial", "trial_seed"]
    for method in successful["method"].drop_duplicates():
        if method == "proposed":
            continue
        baseline = successful.loc[successful["method"] == method]
        paired = proposed[key + list(metrics)].merge(
            baseline[key + list(metrics)],
            on=key,
            suffixes=("_proposed", "_baseline"),
        )
        for setting, group in paired.groupby("setting", sort=False):
            for metric_index, metric in enumerate(metrics):
                differences = (
                    group[f"{metric}_proposed"].to_numpy(dtype=float)
                    - group[f"{metric}_baseline"].to_numpy(dtype=float)
                )
                mean, lower, upper, standard_deviation, n_valid = bootstrap_mean_ci(
                    differences,
                    seed=seed + metric_index * 7919 + len(rows),
                )
                rows.append(
                    {
                        "setting": setting,
                        "baseline": method,
                        "metric": metric,
                        "contrast": "proposed_minus_baseline",
                        "mean_difference": mean,
                        "ci_low": lower,
                        "ci_high": upper,
                        "sd": standard_deviation,
                        "n_pairs": n_valid,
                        "positive_favours_proposed": metric
                        in {"adjacency_f1", "exact_f1"},
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def method_output_to_edge_rows(
    setting: str,
    trial: int,
    trial_seed: int,
    method: str,
    output: MethodOutput,
) -> List[Dict[str, Any]]:
    rows = [
        {
            "setting": setting,
            "trial": trial,
            "trial_seed": trial_seed,
            "method": method,
            "parent": parent,
            "child": child,
            "lag": lag,
            "edge_type": "lagged",
        }
        for parent, child, lag in sorted(output.lagged_edges)
    ]
    rows.extend(
        {
            "setting": setting,
            "trial": trial,
            "trial_seed": trial_seed,
            "method": method,
            "parent": parent,
            "child": child,
            "lag": 0,
            "edge_type": "contemporaneous",
        }
        for parent, child in sorted(output.contemporaneous_edges)
    )
    return rows


def run_method(
    method: str,
    trial: SimulatedTrial,
    config: DGPConfig,
    seed: int,
    args: argparse.Namespace,
    proposed_learner: Optional[ProposedLearner],
) -> MethodOutput:
    if method == "proposed":
        if proposed_learner is None:
            raise SkipMethod("Proposed learner could not be loaded")
        return proposed_learner.run(
            trial.data, trial.node_types, seed, config.L_max, args
        )
    if method == "fixed_lag_1":
        structure, score = fixed_lag_tabu_search(
            trial.data,
            trial.node_types,
            fixed_lag=1,
            penalty_coef=args.penalty_coef,
            max_iter=args.max_iter,
            tabu_length=args.tabu_length,
        )
        return MethodOutput(
            lagged_edges=structure_to_lagged_edges(structure),
            contemporaneous_edges=set(),
            score=score,
            metadata={"selected_lag": 1},
        )
    if method == "fixed_lag_cv":
        return run_fixed_lag_cv(
            trial.data,
            trial.node_types,
            config.L_max,
            args.penalty_coef,
            args.max_iter,
            args.tabu_length,
            args.validation_fraction,
        )
    if method == "pcmci":
        return run_pcmci(trial.data, config, args)
    if method == "dynotears":
        return run_dynotears(trial.data, config, args)
    if method == "lpcmci":
        return run_lpcmci(trial.data, config, args)
    if method == "varlingam":
        return run_varlingam(trial.data, config, seed, args)
    raise ValueError(method)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark variable-lag causal discovery against fixed and external baselines"
    )
    parser.add_argument(
    "--suite",
    choices=("custom", "quick", "reviewer", "multilag"),
    default="custom",
    help=(
        "reviewer supplies the tiered manuscript settings; "
        "multilag runs only the continuous multi-lag setting"
    ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Any of: all, " + ", ".join(ALL_METHODS),
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop on skip/error")
    parser.add_argument("--preflight", action="store_true")

    parser.add_argument(
        "--sweep",
        choices=(
            "none",
            "T",
            "N",
            "density",
            "lagdist",
            "noise",
            "phi",
            "fracbin",
            "conf",
            "mcar",
            "mar",
            "noise_distribution",
        ),
        default="none",
    )
    parser.add_argument("--values", nargs="*", default=[])
    parser.add_argument("--lag-distribution", choices=("uniform", "short", "long"), default="uniform")

    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--L-max", dest="L_max", type=int, default=6)
    parser.add_argument("--p-edge", dest="p_edge", type=float, default=0.15)
    parser.add_argument("--frac-binary", dest="frac_binary", type=float, default=0.5)
    parser.add_argument("--beta-scale", dest="beta_scale", type=float, default=0.8)
    parser.add_argument("--logit-scale", dest="logit_scale", type=float, default=1.2)
    parser.add_argument("--noise-std", dest="noise_std", type=float, default=0.8)
    parser.add_argument(
        "--noise-distribution",
        choices=("gaussian", "laplace", "uniform"),
        default="gaussian",
    )
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--n-confounders", dest="n_confounders", type=int, default=0)
    parser.add_argument("--conf-strength", dest="conf_strength", type=float, default=0.5)
    parser.add_argument("--mcar-rate", dest="mcar_rate", type=float, default=0.0)
    parser.add_argument("--mar-rate", dest="mar_rate", type=float, default=0.0)

    parser.add_argument(
        "--multi-lag-fraction",
        type=float,
        default=None,
        help=(
        "fraction of true adjacencies with multiple lags; "
        "defaults to 0.5 for the multilag suite"
        ),
    )

    parser.add_argument(
        "--lags-per-multi-edge",
        type=int,
        default=2,
        help="number of distinct true lags on each selected multi-lag adjacency",
    )
    parser.add_argument("--learner-file", default=None)
    parser.add_argument("--parallel-learner-file", default=None)
    parser.add_argument("--proposed-parallel", action="store_true")
    parser.add_argument("--proposed-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--penalty-coef", type=float, default=0.5)
    parser.add_argument("--tabu-length", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.2)

    parser.add_argument("--external-verbosity", type=int, default=0)
    parser.add_argument("--allow-assumption-violations", action="store_true")
    parser.add_argument("--lpcmci-all-continuous", action="store_true")
    parser.add_argument(
        "--lpcmci-include-contemporaneous",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pcmci-pc-alpha", type=float, default=0.05)
    parser.add_argument("--pcmci-alpha", type=float, default=0.05)
    parser.add_argument("--pcmci-fdr", default="fdr_bh")
    parser.add_argument("--lpcmci-alpha", type=float, default=0.05)
    parser.add_argument("--lpcmci-max-p-global", type=int, default=None, help="Maximum conditioning-set size for all LPCMCI phases",)
    parser.add_argument("--lpcmci-max-p-non-ancestral", type=int, default=None, help="Maximum conditioning-set size in LPCMCI's non-ancestral phase",)
    parser.add_argument("--lpcmci-max-q-global", type=int, default=None, help="Maximum conditioning sets tested per pair and cardinality",)
    parser.add_argument("--dynotears-lambda-w", type=float, default=0.1)
    parser.add_argument("--dynotears-lambda-a", type=float, default=0.1)
    parser.add_argument("--dynotears-max-iter", type=int, default=100)
    parser.add_argument("--dynotears-h-tol", type=float, default=1e-8)
    parser.add_argument("--dynotears-weight-threshold", type=float, default=0.01)
    parser.add_argument("--varlingam-threshold", type=float, default=0.05)
    parser.add_argument(
        "--varlingam-prune", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("trials must be at least 1")
    if args.N < 1 or args.T < 10 or args.L_max < 1:
        raise ValueError("Require N >= 1, T >= 10, and L_max >= 1")
    if args.L_max >= args.T - 2:
        raise ValueError("L_max must be smaller than T - 2")
    if args.n_confounders < 0:
        raise ValueError("n_confounders cannot be negative")
    for name in ("mcar_rate", "mar_rate"):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must lie in [0, 1)")
    if args.multi_lag_fraction is not None:
        if not 0.0 <= args.multi_lag_fraction <= 1.0:
            raise ValueError(
                "multi_lag_fraction must lie in [0, 1]"
            )
    multi_lag_requested = (
        args.suite == "multilag"
        or (
            args.multi_lag_fraction is not None
            and args.multi_lag_fraction > 0.0
        )
    )
    if args.suite == "multilag" and args.multi_lag_fraction == 0.0:
        raise ValueError(
            "--multi-lag-fraction must be greater than zero "
            "for --suite multilag"
    )
    if multi_lag_requested:
        if args.N < 2 or args.p_edge <= 0.0:
            raise ValueError(
                "A multi-lag setting requires N >= 2 and p_edge > 0"
            )
        if args.L_max < 2:
            raise ValueError(
                "A multi-lag setting requires L_max >= 2"
            )
        if not 2 <= args.lags_per_multi_edge <= args.L_max:
            raise ValueError(
                "lags_per_multi_edge must lie in [2, L_max]"
            )
    


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    methods = parse_methods(args.methods)

    if args.preflight:
        print(json.dumps(dependency_report(args), indent=2, sort_keys=True))
        return 0

    settings = build_settings(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_directory = Path(
        args.output_dir or f"benchmark_results_{timestamp}"
    ).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    data_directory = output_directory / "simulated_data"
    if args.save_data:
        data_directory.mkdir(exist_ok=True)

    proposed_learner: Optional[ProposedLearner] = None
    proposed_load_error: Optional[str] = None
    if "proposed" in methods:
        try:
            proposed_learner = ProposedLearner(args)
        except Exception as exc:
            proposed_load_error = str(exc)
            if args.strict:
                raise

    trial_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    
    completed_jobs = 0
    for setting_index, (setting, config, lag_distribution) in enumerate(settings):
        print(
            f"\n[{setting}] N={config.N}, T={config.T}, L_max={config.L_max}, "
            f"binary={config.frac_binary:.2f}, confounders={config.n_confounders}, "
            f"noise={config.noise_distribution}, lags={lag_distribution}"
        )
        for trial_index in range(args.trials):
            trial_seed = int(
                np.random.SeedSequence(
                    [args.seed, setting_index, trial_index]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            simulated = simulate_trial(config, lag_distribution, trial_seed)
            truth_output = MethodOutput(
                lagged_edges=set(simulated.true_edges),
                contemporaneous_edges={
                    directed
                    for pair in simulated.confounded_pairs
                    for directed in (pair, (pair[1], pair[0]))
                },
                metadata={"truth": True},
            )
            edge_rows.extend(
                method_output_to_edge_rows(
                    setting, trial_index, trial_seed, "truth", truth_output
                )
            )
            if args.save_data:
                np.savez_compressed(
                    data_directory / f"{setting}_trial_{trial_index:03d}.npz",
                    data=simulated.data,
                    node_types=np.array(
                        [simulated.node_types[node] for node in range(config.N)]
                    ),
                    true_edges=np.array(sorted(simulated.true_edges), dtype=int).reshape(-1, 3),
                )

            for method in methods:
                completed_jobs += 1
                base_row: Dict[str, Any] = {
                    "setting": setting,
                    "lag_distribution": lag_distribution,
                    "trial": trial_index,
                    "trial_seed": trial_seed,
                    "method": method,
                    **asdict(config),
                }
                reason = method_eligibility(method, config, args)
                if method == "proposed" and proposed_learner is None:
                    reason = "proposed learner unavailable: " + str(proposed_load_error)
                if reason is not None:
                    if args.strict:
                        raise SkipMethod(f"{method}: {reason}")
                    trial_rows.append(
                        {**base_row, "status": "skipped", "error": reason}
                    )
                    print(
                        f"  trial {trial_index + 1:02d}/{args.trials} {method:<14} SKIP: {reason}"
                    )
                    continue

                started = time.perf_counter()
                try:
                    output = run_method(
                        method,
                        simulated,
                        config,
                        trial_seed + 12345,
                        args,
                        proposed_learner,
                    )
                    runtime = time.perf_counter() - started
                    metrics = evaluate_method(simulated, output, runtime, config)
                    trial_rows.append(
                        {**base_row, "status": "ok", "error": "", **metrics}
                    )
                    edge_rows.extend(
                        method_output_to_edge_rows(
                            setting, trial_index, trial_seed, method, output
                        )
                    )
                    print(
                        f"  trial {trial_index + 1:02d}/{args.trials} {method:<14} "
                        f"adjF1={metrics['adjacency_f1']:.3f} "
                        f"exactF1={metrics['exact_f1']:.3f} "
                        f"time={runtime:.2f}s"
                    )
                except SkipMethod as exc:
                    if args.strict:
                        raise
                    trial_rows.append(
                        {**base_row, "status": "skipped", "error": str(exc)}
                    )
                    print(
                        f"  trial {trial_index + 1:02d}/{args.trials} {method:<14} SKIP: {exc}"
                    )
                except Exception as exc:
                    if args.strict:
                        raise
                    trial_rows.append(
                        {
                            **base_row,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(
                        f"  trial {trial_index + 1:02d}/{args.trials} {method:<14} "
                        f"ERROR: {type(exc).__name__}: {exc}"
                    )

    trials_frame = pd.DataFrame(trial_rows)
    edges_frame = pd.DataFrame(
        edge_rows,
        columns=[
            "setting",
            "trial",
            "trial_seed",
            "method",
            "parent",
            "child",
            "lag",
            "edge_type",
        ],
    )
    summary_frame = make_summary(trials_frame, args.seed + 100_000)
    paired_frame = make_paired_differences(trials_frame, args.seed + 200_000)

    trials_path = output_directory / "benchmark_trials.csv"
    edges_path = output_directory / "benchmark_edges.csv"
    summary_path = output_directory / "benchmark_summary.csv"
    paired_path = output_directory / "benchmark_paired_differences.csv"
    manifest_path = output_directory / "benchmark_manifest.json"
    trials_frame.to_csv(trials_path, index=False)
    edges_frame.to_csv(edges_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    paired_frame.to_csv(paired_path, index=False)

    manifest = {
        "created_utc": timestamp,
        "command": [sys.executable, *sys.argv],
        "arguments": vars(args),
        "methods": methods,
        "settings": [
            {
                "name": name,
                "lag_distribution": lag_distribution,
                "config": asdict(config),
            }
            for name, config, lag_distribution in settings
        ],
        "dependencies": dependency_report(args),
        "outputs": {
            "trials": str(trials_path),
            "edges": str(edges_path),
            "summary": str(summary_path),
            "paired_differences": str(paired_path),
        },
        "metric_notes": {
            "adjacency": "directed (parent, child), collapsed over positive lags",
            "exact": "directed (parent, child, lag)",
            "exact_shd": "set symmetric difference; a wrong lag costs one deletion and one addition",
            "lag_mae_conditional": "minimum absolute lag error among predicted lags, conditional on matching an ordered adjacency",
            "confidence_intervals": "95% percentile bootstrap intervals over paired simulation trials",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    status_counts = trials_frame["status"].value_counts(dropna=False).to_dict()
    print("\nCompleted benchmark")
    print(f"  method-runs considered: {completed_jobs}")
    print(f"  statuses: {status_counts}")
    print(f"  trials: {trials_path}")
    print(f"  summary: {summary_path}")
    print(f"  paired differences: {paired_path}")
    print(f"  edge audit: {edges_path}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
