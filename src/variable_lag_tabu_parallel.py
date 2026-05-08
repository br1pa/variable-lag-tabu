"""variable_lag_tabu_parallel.py

Parallel neighbourhood evaluation for variable-lag Tabu search on multivariate time series.

This file is a new implementation of the provided `variable_lag_tabu.py`.
It imports shared GLM/scoring utilities from that module and keeps only the
parallel/decomposable search changes here.

Key changes vs. the original file
--------------------------------
1) Decomposable scoring: the total score is maintained as a sum of local node scores.
   
2) Parallel neighbourhood evaluation: within each greedy/Tabu iteration, candidate moves
   are evaluated in parallel. The time-series array must be placed in shared memory.

Notes
-----
* This implementation follows the paper's time-unrolled acyclicity guarantee: edges are only
  from the past to the future (lag >= 1). Therefore, the induced unrolled graph is always acyclic.
* The compact lagged graph may contain feedback cycles across variables (at different lags).
"""

from __future__ import annotations

import copy
import os
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from multiprocessing import shared_memory
except Exception:
    shared_memory = None

from concurrent.futures import ProcessPoolExecutor

from variable_lag_tabu import (
            _sigmoid,
            _log1pexp,
            _logit_loglik,
            _logistic_irls,
            compute_score_mixed as _base_compute_score_mixed,
        )

# ============================
#  Decomposable mixed scorer
# ============================

Parents = List[Tuple[int, int]]  # [(parent, lag>=1), ...]
Structure = Dict[int, Parents]   # {child: parents}
Move = Tuple[str, int, int, Optional[int], Optional[int]] # ('add'|'remove'|'reverse'|'lag', u, v, newLag, oldLag)

INVALID_LOCAL_SCORE = -1e12
INVALID_GLOBAL_SCORE = -1e9

def _normalise_parents(parents: Parents, L_max: int) -> Parents:
    """Clamp lags and drop duplicate parents while preserving first-seen order.

    The original implementation preserves parent order because edges are appended,
    removed, and lag-updated in place. We keep that ordering here so lag-tuning
    and tie-breaking can mirror the original implementation.
    """
    out: Parents = []
    seen = set()
    for p, l in parents:
        p = int(p)
        if p in seen:
            continue
        seen.add(p)
        out.append((p, max(1, min(int(l), int(L_max)))))
    return out


@dataclass
class MixedLagScorer:
    """Decomposable BIC + lag penalty score for mixed (continuous/binary) time-series nodes."""

    data: np.ndarray                 # shape (T, N)
    node_types: List[str]            # length N: 'continuous'|'binary'
    penalty_coef: float = 0.5
    L_max: int = 6
    irls_max_iter: int = 50
    irls_tol: float = 1e-6
    irls_ridge: float = 1e-8
    cache_max: int = 50_000          # simple capped cache

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 2:
            raise ValueError("data must be 2D array (T, N)")
        self.T, self.N = self.data.shape
        if len(self.node_types) != self.N:
            raise ValueError("node_types length must match data.shape[1]")
        for t in self.node_types:
            if t not in ("continuous", "binary"):
                raise NotImplementedError("Only 'continuous' and 'binary' supported")
        self._cache: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], float] = {}

    def _cache_get(self, key: Tuple[int, Tuple[Tuple[int, int], ...]]) -> Optional[float]:
        return self._cache.get(key)

    def _cache_put(self, key: Tuple[int, Tuple[Tuple[int, int], ...]], val: float) -> None:
        if len(self._cache) >= self.cache_max:
            for k in list(self._cache.keys())[: max(1, self.cache_max // 10)]:
                self._cache.pop(k, None)
        self._cache[key] = val

    def local_score(self, child: int, parents: Parents) -> float:
        """Compute the score for one child node, based on its parent nodes and their lags.

        Returns: 2*loglik - p*log(n_eff) - penalty_coef*sum(max(0,lag-1)).
        Returns INVALID_LOCAL_SCORE when the corresponding node-local fit would
        make the scorer return INVALID_GLOBAL_SCORE.
        """
        parents = _normalise_parents(parents, self.L_max)
        key = (int(child), tuple(parents))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        Lj = max((lag for _, lag in parents), default=0)
        n_eff = self.T - Lj
        if n_eff <= 1:
            self._cache_put(key, INVALID_LOCAL_SCORE)
            return INVALID_LOCAL_SCORE

        y_raw = self.data[Lj:, child]

        X_cols = []
        for p, lag in parents:
            p_series = self.data[Lj - lag : self.T - lag, p]
            if self.node_types[p] == "binary":
                uniq = np.unique(p_series)
                if uniq.size == 1:
                    col = (p_series == uniq[0]).astype(float)
                elif uniq.size == 2:
                    col = (p_series == uniq.max()).astype(float)
                else:
                    col = p_series.astype(float)
            else:
                col = p_series.astype(float)
            X_cols.append(col.reshape(-1, 1))

        if X_cols:
            X = np.hstack(X_cols)
            X = np.hstack([np.ones((n_eff, 1)), X])
        else:
            X = np.ones((n_eff, 1))

        child_type = self.node_types[child]
        p_j = X.shape[1]

        if child_type == "continuous":
            beta, *_ = np.linalg.lstsq(X, y_raw, rcond=None)
            resid = y_raw - X @ beta
            RSS = float(np.sum(resid ** 2))
            sigma2 = RSS / n_eff
            if sigma2 <= 0:
                sigma2 = 1e-9
            loglik = -0.5 * n_eff * (np.log(2 * np.pi * sigma2) + 1.0)
        else:
            uniq_y = np.unique(y_raw)
            if uniq_y.size == 1:
                y = (y_raw == uniq_y.max()).astype(float)
            elif uniq_y.size == 2:
                y = (y_raw == uniq_y.max()).astype(float)
            else:
                raise ValueError(f"Child {child} declared 'binary' but has >2 unique values")
            _, loglik = _logistic_irls(
                X,
                y,
                max_iter=self.irls_max_iter,
                tol=self.irls_tol,
                ridge=self.irls_ridge,
            )

        bic_term = 2.0 * float(loglik) - float(p_j) * np.log(max(n_eff, 2))
        lag_pen = sum(max(0, lag - 1) for _, lag in parents)
        val = bic_term - self.penalty_coef * float(lag_pen)
        self._cache_put(key, val)
        return val

    def score(self, struct: Structure) -> float:
        total = 0.0
        for j in range(self.N):
            local = self.local_score(j, struct.get(j, []))
            if local <= INVALID_LOCAL_SCORE:
                return INVALID_GLOBAL_SCORE
            total += local
        return float(total)

# ============================
#  Greedy lag tuning (local)
# ============================

def greedy_lag_tune_for_child(
    scorer: MixedLagScorer,
    child: int,
    parents: Parents,
) -> Tuple[Parents, float]:
    """Mirror the original implementation's per-child lag optimisation.

    Returns (tuned_parents, tuned_local_score).
    """
    parents = _normalise_parents(parents, scorer.L_max)
    if not parents:
        return parents, scorer.local_score(child, parents)

    improved = True
    while improved:
        improved = False
        for idx, (p, lag) in enumerate(list(parents)):
            base_s = scorer.local_score(child, parents)
            best_s = base_s
            best_l = lag

            if lag < scorer.L_max:
                cand = parents.copy()
                cand[idx] = (p, lag + 1)
                cand = _normalise_parents(cand, scorer.L_max)
                s_plus = scorer.local_score(child, cand)
                if s_plus > best_s:
                    best_s = s_plus
                    best_l = lag + 1

            if lag > 1:
                cand = parents.copy()
                cand[idx] = (p, lag - 1)
                cand = _normalise_parents(cand, scorer.L_max)
                s_minus = scorer.local_score(child, cand)
                if s_minus > best_s:
                    best_s = s_minus
                    best_l = lag - 1

            if best_l != lag:
                parents[idx] = (p, best_l)
                parents = _normalise_parents(parents, scorer.L_max)
                improved = True

    return parents, scorer.local_score(child, parents)

# ============================
#  Move enumeration / helpers
# ============================

def enumerate_moves(struct: Structure, N: int, L_max: int) -> List[Move]:
    """Enumerate moves in the same order as the original implementation.

    The original order is:
      1) for each (u, v): add, remove, reverse
      2) Once all add/remove/reverse moves have been listed, go through the existing edges one by one and add lag-increase / lag-decrease moves in the order those edges appear in the structure
    """
    lag_mat = np.zeros((N, N), dtype=int)
    for child in range(N):
        for p, lag in struct.get(child, []):
            lag_mat[p, child] = int(lag)

    moves: List[Move] = []

    for u in range(N):
        for v in range(N):
            if lag_mat[u, v] == 0:
                moves.append(("add", u, v, 1, None))

            if lag_mat[u, v] > 0:
                moves.append(("remove", u, v, None, int(lag_mat[u, v])))

            if u != v and lag_mat[u, v] > 0 and lag_mat[v, u] == 0:
                moves.append(("reverse", u, v, 1, int(lag_mat[u, v])))

    for child in range(N):
        for p, lag in list(struct.get(child, [])):
            if lag < L_max:
                moves.append(("lag", p, child, lag + 1, lag))
            if lag > 1:
                moves.append(("lag", p, child, lag - 1, lag))

    return moves


def _get_lag(parents: Parents, parent: int) -> Optional[int]:
    for p, lag in parents:
        if p == parent:
            return int(lag)
    return None


def _remove_parent(parents: Parents, parent: int) -> Parents:
    return [(p, lag) for (p, lag) in parents if p != parent]


def _set_parent_lag(parents: Parents, parent: int, lag: int, L_max: int) -> Parents:
    lag = max(1, min(int(lag), int(L_max)))
    out: Parents = []
    found = False
    for p, l in parents:
        if p == parent:
            out.append((p, lag))
            found = True
        else:
            out.append((p, int(l)))
    if not found:
        out.append((int(parent), lag))
    return _normalise_parents(out, L_max)


def _copy_struct(struct: Structure, N: int) -> Structure:
    return {j: [(p, lag) for (p, lag) in struct.get(j, [])] for j in range(N)}

# ============================
#  Candidate evaluation
# ============================

@dataclass
class CandidateResult:
    score: float
    move: Move
    updated_children: Dict[int, Parents]
    updated_locals: Dict[int, float]


def evaluate_move_decomposable(
    scorer: MixedLagScorer,
    current_struct: Structure,
    current_score: float,
    local_scores: np.ndarray,
    move: Move,
    tune_after_lag: bool = False,
    tune_reverse_children: str = "new_child_only",
) -> CandidateResult:
    """Test one possible change to the current graph, and score it efficiently by only recomputing the parts that changed.

    This parallel/decomposable version is trying to behave the same way as the original.
    In particular, reverse moves tune only the node that gains the reversed edge by default.
    """
    act, u, v, newLag, oldLag = move
    L_max = scorer.L_max

    def _invalid() -> CandidateResult:
        return CandidateResult(
            score=INVALID_GLOBAL_SCORE,
            move=move,
            updated_children={},
            updated_locals={},
        )

    if act == "add":
        child = v
        parents = list(current_struct.get(child, []))
        if any(p == u for p, _ in parents):
            return _invalid()
        parents.append((u, 1))
        parents = _normalise_parents(parents, L_max)
        parents, s_local = greedy_lag_tune_for_child(scorer, child, parents)
        if s_local <= INVALID_LOCAL_SCORE:
            return _invalid()
        final_lag = _get_lag(parents, u)
        move_final: Move = ("add", u, v, final_lag if final_lag is not None else 1, None)
        cand_score = current_score - float(local_scores[child]) + s_local
        return CandidateResult(cand_score, move_final, {child: parents}, {child: s_local})

    if act == "remove":
        child = v
        parents = list(current_struct.get(child, []))
        if not any(p == u for p, _ in parents):
            return _invalid()
        parents2 = _normalise_parents(_remove_parent(parents, u), L_max)
        s_local = scorer.local_score(child, parents2)
        if s_local <= INVALID_LOCAL_SCORE:
            return _invalid()
        cur_lag = _get_lag(parents, u)
        move_final = ("remove", u, v, None, cur_lag)
        cand_score = current_score - float(local_scores[child]) + s_local
        return CandidateResult(cand_score, move_final, {child: parents2}, {child: s_local})

    if act == "lag":
        child = v
        parents = list(current_struct.get(child, []))
        if not any(p == u for p, _ in parents):
            return _invalid()
        cur_lag = _get_lag(parents, u)
        parents2 = _set_parent_lag(parents, u, int(newLag), L_max)
        if tune_after_lag:
            parents2, s_local = greedy_lag_tune_for_child(scorer, child, parents2)
        else:
            s_local = scorer.local_score(child, parents2)
        if s_local <= INVALID_LOCAL_SCORE:
            return _invalid()
        final_lag = _get_lag(parents2, u)
        move_final = ("lag", u, v, final_lag, cur_lag)
        cand_score = current_score - float(local_scores[child]) + s_local
        return CandidateResult(cand_score, move_final, {child: parents2}, {child: s_local})

    if act == "reverse":
        if u == v:
            return _invalid()

        child1 = v
        parents1 = list(current_struct.get(child1, []))
        if not any(p == u for p, _ in parents1):
            return _invalid()
        parents1b = _normalise_parents(_remove_parent(parents1, u), L_max)
        s1 = scorer.local_score(child1, parents1b)
        if s1 <= INVALID_LOCAL_SCORE:
            return _invalid()

        child2 = u
        parents2 = list(current_struct.get(child2, []))
        if any(p == v for p, _ in parents2):
            return _invalid()
        parents2.append((v, 1))
        parents2 = _normalise_parents(parents2, L_max)

        if tune_reverse_children == "both":
            parents1b, s1 = greedy_lag_tune_for_child(scorer, child1, parents1b)
            parents2, s2 = greedy_lag_tune_for_child(scorer, child2, parents2)
        else:
            s1 = scorer.local_score(child1, parents1b)
            parents2, s2 = greedy_lag_tune_for_child(scorer, child2, parents2)

        if s1 <= INVALID_LOCAL_SCORE or s2 <= INVALID_LOCAL_SCORE:
            return _invalid()

        old_lag = _get_lag(parents1, u)
        new_lag = _get_lag(parents2, v)
        move_final: Move = ("reverse", u, v, new_lag if new_lag is not None else 1, old_lag)
        cand_score = (
            current_score
            - float(local_scores[child1])
            - float(local_scores[child2])
            + s1
            + s2
        )
        return CandidateResult(
            cand_score,
            move_final,
            {child1: parents1b, child2: parents2},
            {child1: s1, child2: s2},
        )

    return _invalid()


def apply_candidate(
    struct: Structure,
    local_scores: np.ndarray,
    cand: CandidateResult,
    N: int,
) -> Tuple[Structure, np.ndarray, float]:
    """Take the chosen move changes, apply them to the graph, and keep the graph stored in the same full dictionary format as the original"""
    struct2: Structure = _copy_struct(struct, N)
    local2 = local_scores.copy()

    for child, parents in cand.updated_children.items():
        struct2[child] = list(parents)
    for child, s_local in cand.updated_locals.items():
        local2[child] = float(s_local)

    total = float(np.sum(local2))
    return struct2, local2, total


# ============================
#  Parallel worker 
# ============================

_WORKER_SCORER: Optional[MixedLagScorer] = None
_WORKER_DATA_SHM = None
_WORKER_DATA_ARR: Optional[np.ndarray] = None


def _worker_init(
    shm_name: Optional[str],
    shape: Tuple[int, int],
    dtype_str: str,
    node_types: List[str],
    penalty_coef: float,
    L_max: int,
    irls_max_iter: int,
    irls_tol: float,
    irls_ridge: float,
) -> None:
    global _WORKER_SCORER, _WORKER_DATA_SHM, _WORKER_DATA_ARR
    if shm_name is None:
        raise RuntimeError("Shared memory name is None; parallel worker requires shared memory")
    if shared_memory is None:
        raise RuntimeError("multiprocessing.shared_memory not available in this Python version")
    _WORKER_DATA_SHM = shared_memory.SharedMemory(name=shm_name)
    _WORKER_DATA_ARR = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_WORKER_DATA_SHM.buf)
    _WORKER_SCORER = MixedLagScorer(
        data=_WORKER_DATA_ARR,
        node_types=node_types,
        penalty_coef=penalty_coef,
        L_max=L_max,
        irls_max_iter=irls_max_iter,
        irls_tol=irls_tol,
        irls_ridge=irls_ridge,
    )


def _worker_eval(args: Tuple[Structure, float, np.ndarray, Move, bool, str]) -> CandidateResult:
    current_struct, current_score, local_scores, move, tune_after_lag, tune_reverse_children = args
    if _WORKER_SCORER is None:
        raise RuntimeError("Worker scorer not initialised")
    return evaluate_move_decomposable(
        scorer=_WORKER_SCORER,
        current_struct=current_struct,
        current_score=current_score,
        local_scores=local_scores,
        move=move,
        tune_after_lag=tune_after_lag,
        tune_reverse_children=tune_reverse_children,
    )


# ============================
#  Main algorithm
# ============================

def tabu_search_variable_lag_parallel(
    data: np.ndarray,
    node_types: Dict[int, str] | List[str],
    L_max: Optional[int] = 6,
    penalty_coef: float = 0.5,
    greedy_init: bool = True,
    greedy_max_rounds: Optional[int] = None,
    tabu_max_iter: int = 200,
    tabu_length: int = 25,
    tune_after_lag: bool = False,
    tune_reverse_children: str = "new_child_only",  # 'both'|'new_child_only'
    n_jobs: int = 1,
    use_shared_memory: bool = True,
    mp_start_method: Optional[str] = None,
    irls_max_iter: int = 50,
    irls_tol: float = 1e-6,
    irls_ridge: float = 1e-8,
) -> Tuple[Structure, float]:
    """Variable-lag Tabu search with optional parallel neighbourhood evaluation.

    The default settings mirror the original implementation.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array (T, N)")
    T, N = data.shape
    if L_max is None:
        L_max = max(1, min(10, T - 2))
    L_max = int(L_max)

    if isinstance(node_types, dict):
        node_types_list = [node_types[j] for j in range(N)]
    else:
        node_types_list = list(node_types)
    if len(node_types_list) != N:
        raise ValueError("node_types must cover all N nodes")

    scorer = MixedLagScorer(
        data=data,
        node_types=node_types_list,
        penalty_coef=penalty_coef,
        L_max=L_max,
        irls_max_iter=irls_max_iter,
        irls_tol=irls_tol,
        irls_ridge=irls_ridge,
    )

    current: Structure = {j: [] for j in range(N)}
    local_scores = np.array([scorer.local_score(j, []) for j in range(N)], dtype=float)
    current_score = float(np.sum(local_scores))
    best_struct: Structure = _copy_struct(current, N)
    best_score = current_score

    tabu_list: List[Move] = []

    executor: Optional[ProcessPoolExecutor] = None
    shm = None
    shm_name = None

    if n_jobs is None:
        n_jobs = 1
    n_jobs = max(1, int(n_jobs))

    if n_jobs > 1:
        if mp_start_method is None:
            mp_start_method = "fork" if os.name == "posix" else "spawn"
        try:
            ctx = mp.get_context(mp_start_method)
        except ValueError:
            ctx = mp.get_context("spawn")

        if use_shared_memory:
            if shared_memory is None:
                raise RuntimeError("Shared memory not available; set use_shared_memory=False")
            shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shm_name = shm.name
            shm_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            shm_arr[:] = data
        else:
            raise RuntimeError(
                "For this implementation, parallel mode requires use_shared_memory=True. "
            )

        executor = ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                shm_name,
                data.shape,
                str(data.dtype),
                node_types_list,
                penalty_coef,
                L_max,
                irls_max_iter,
                irls_tol,
                irls_ridge,
            ),
        )

    def eval_all_moves(moves: List[Move]) -> List[CandidateResult]:
        if not moves:
            return []
        if executor is None:
            return [
                evaluate_move_decomposable(
                    scorer=scorer,
                    current_struct=current,
                    current_score=current_score,
                    local_scores=local_scores,
                    move=mv,
                    tune_after_lag=tune_after_lag,
                    tune_reverse_children=tune_reverse_children,
                )
                for mv in moves
            ]

        args_iter = (
            (current, current_score, local_scores, mv, tune_after_lag, tune_reverse_children)
            for mv in moves
        )
        return list(executor.map(_worker_eval, args_iter, chunksize=32))

    try:
        if greedy_init:
            greedy_rounds = 0
            while True:
                if greedy_max_rounds is not None and greedy_rounds >= int(greedy_max_rounds):
                    break
                greedy_rounds += 1

                improved = False
                best_cand: Optional[CandidateResult] = None
                best_move_score = current_score

                moves = enumerate_moves(current, N, L_max)
                cands = eval_all_moves(moves)

                for c in cands:
                    if c.score > best_move_score:
                        best_move_score = c.score
                        best_cand = c

                if best_cand is not None and best_move_score > current_score:
                    current, local_scores, current_score = apply_candidate(current, local_scores, best_cand, N)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_struct = _copy_struct(current, N)

                if not improved:
                    break

        for _ in range(tabu_max_iter):
            moves = enumerate_moves(current, N, L_max)
            cands = eval_all_moves(moves)

            best_adm: Optional[CandidateResult] = None
            best_adm_score = INVALID_GLOBAL_SCORE
            for c in cands:
                if ((c.move not in tabu_list) or (c.score > best_score)) and (c.score > best_adm_score):
                    best_adm = c
                    best_adm_score = c.score

            if best_adm is None:
                break

            current, local_scores, current_score = apply_candidate(current, local_scores, best_adm, N)

            if len(tabu_list) >= tabu_length:
                tabu_list.pop(0)
            act, u, v, newLag, oldLag = best_adm.move
            if act == "add":
                inverse: Optional[Move] = ("remove", u, v, None, newLag)
            elif act == "remove":
                inverse = ("add", u, v, 1, None)
            elif act == "reverse":
                inverse = ("reverse", v, u, oldLag if oldLag is not None else 1, newLag)
            elif act == "lag":
                inverse = ("lag", u, v, oldLag, newLag)
            else:
                inverse = None
            if inverse is not None:
                tabu_list.append(inverse)

            if current_score > best_score:
                best_score = current_score
                best_struct = _copy_struct(current, N)

    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if shm is not None:
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

    return best_struct, float(best_score)


if __name__ == "__main__":
    pass