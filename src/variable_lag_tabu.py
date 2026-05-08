import numpy as np

def _sigmoid(z):
    
    """
    Computes logistic function element-wise for a NumPy array (or scalar)
    """
    
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def _log1pexp(x):
    
    """
    Compute log(1 + exp(x)) element-wise on x.
    """
    
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def _logit_loglik(eta, y):
    
    """
    Compute the log-likelihood for a binary logistic model
    y in {0,1}, eta = X @ beta
    log p(y|eta) = sum(y*eta - log(1 + exp(eta)))
    """
    
    return float(np.sum(y * eta - _log1pexp(eta)))

def _logistic_irls(X, y, max_iter=50, tol=1e-6, ridge=1e-8):
    """
    Iteratively Reweighted Least Squares (IRLS) for binary logistic regression.
    Returns (beta, loglik).
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = _sigmoid(eta)
        W = mu * (1.0 - mu)
        W = np.clip(W, 1e-9, None)
        z = eta + (y - mu) / W
        # Solve (X^T W X + ridge I) beta = X^T W z
        # via weighted least squares on sqrt(W)*X, sqrt(W)*z
        sW = np.sqrt(W)
        Xw = X * sW[:, None]
        zw = z * sW
        A = Xw.T @ Xw + ridge * np.eye(p)
        b = Xw.T @ zw
        beta_new = np.linalg.solve(A, b)
        if np.linalg.norm(beta_new - beta) < tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    ll = _logit_loglik(eta, y)
    return beta, ll

def compute_score_mixed(data, structure, node_types, penalty_coef=0.5, L_max=None):
    """
    Decomposable score for mixed data.
    Parents can be continuous or binary (auto-handled).
    Children:
      - 'continuous': linear Gaussian (OLS)
      - 'binary': logistic regression (IRLS)

    Args
    ----
    data : np.ndarray of shape (T, n_vars)
    structure : dict {child: [(parent, lag), ...], ...} with lag >= 1
    node_types : dict {j: 'continuous'|'binary'} for each variable j
    penalty_coef : float, coefficient for lag penalty sum max(0, lag-1)
    L_max : int or None, max allowed lag (defensively clamps if provided)

    Returns
    -------
    score : float  (higher is better)
    """
    T, n_vars = data.shape

    # validate node_types
    for j in range(n_vars):
        if j not in node_types:
            raise ValueError(f"node_types missing entry for node {j}")
        if node_types[j] not in ('continuous', 'binary'):
            raise NotImplementedError("Only 'continuous' and 'binary' children supported in this version.")

    bic_total = 0.0  # sum over nodes: 2*loglik_j - p_j * log(n_j)

    for j in range(n_vars):
        parents = structure.get(j, [])
        if L_max is not None and parents:
            parents = [(p, min(int(lag), L_max)) for (p, lag) in parents]

        # Node-specific alignment lag
        Lj = max([lag for (_, lag) in parents], default=0)
        n_eff = T - Lj
        if n_eff <= 1:
            return -1e9

        # Child response
        y_raw = data[Lj:, j]
        child_type = node_types[j]

        X_cols = []
        # For each parent, get aligned series to y_raw
        for (p, lag) in parents:
            p_series = data[Lj - lag : T - lag, p]

            parent_type = node_types[p]
            if parent_type == 'binary':
                # Map two unique values to {0,1}
                uniq = np.unique(p_series)
                if uniq.size == 1:
                    # constant parent: still include as a constant column
                    col = (p_series == uniq[0]).astype(float)  # all ones
                elif uniq.size == 2:
                    # map smaller to 0, larger to 1
                    col = (p_series == uniq.max()).astype(float)
                else:
                    # if mislabeled as binary but has >2 levels, fall back to simple numeric encoding
                    col = p_series.astype(float)
                X_cols.append(col.reshape(-1, 1))
            else:
                # 'continuous' parent: just pass through as numeric
                X_cols.append(p_series.reshape(-1, 1))

        # Stack or start empty
        if X_cols:
            X = np.hstack(X_cols)
            # add intercept
            X = np.hstack([np.ones((n_eff, 1)), X])
        else:
            # if no parents, intercept-only model
            X = np.ones((n_eff, 1))

        # ----- Fit child model & compute local BIC -----
        if child_type == 'continuous':
            # OLS Gaussian
            beta, *_ = np.linalg.lstsq(X, y_raw, rcond=None)
            resid = y_raw - X @ beta
            RSS = float(np.sum(resid**2))
            sigma2 = RSS / n_eff
            if sigma2 <= 0:
                sigma2 = 1e-9
            loglik_j = -0.5 * n_eff * (np.log(2 * np.pi * sigma2) + 1.0)
            p_j = X.shape[1]  # intercept + slopes
            bic_total += (2.0 * loglik_j - p_j * np.log(max(n_eff, 2)))

        elif child_type == 'binary':
            # Map y to {0,1} if needed
            uniq_y = np.unique(y_raw)
            if uniq_y.size == 1:
                # model fits a constant
                y = (y_raw == uniq_y.max()).astype(float)
            elif uniq_y.size == 2:
                y = (y_raw == uniq_y.max()).astype(float)
            else:
                raise ValueError(f"Child {j} declared 'binary' but has >2 unique values.")

            beta, loglik_j = _logistic_irls(X, y)
            p_j = X.shape[1]
            bic_total += (2.0 * loglik_j - p_j * np.log(max(n_eff, 2)))

        else:
            raise NotImplementedError("Only 'continuous' and 'binary' children supported.")

    # -------- Lag penalty --------
    lag_penalty = 0
    for child, parents in structure.items():
        for (_, lag) in parents:
            if L_max is not None:
                lag = min(int(lag), L_max)
            if lag > 1:
                lag_penalty += (lag - 1)

    return bic_total - penalty_coef * lag_penalty

def tabu_search_greedy_lag(
    data,
    node_types,
    max_iter=100,
    tabu_length=10,
    penalty_coef=0.5,
    L_max=None,
    seed=None
):
    """
    Tabu search + greedy lag tuning for mixed data.
    Parents may be continuous or binary; children may be continuous or binary.

    Note:
      Autoregressive (self-lag) edges are allowed, i.e. u==v edges such as
      X_{t-1} -> X_t. These correspond to (u -> u, lag>=1) in the compact
      representation used here.

    Args
    ----
    data : np.ndarray (T, n_vars)
    node_types : dict {j: 'continuous'|'binary'}
    max_iter : int, Tabu iterations after greedy climb
    tabu_length : int, length of Tabu list
    penalty_coef : float, lag penalty coefficient
    L_max : int or None, maximum lag (autoset if None)
    seed : optional RNG seed

    Returns
    -------
    (best_structure, best_score)
    """
    rng = np.random.default_rng(seed)
    T, n_vars = data.shape
    if L_max is None:
        L_max = max(1, min(10, T - 2))

    def copy_struct(struct):
        # When the search makes a “candidate” graph, it doesn’t accidentally modify the current graph too
        return {j: [(p, lag) for (p, lag) in struct.get(j, [])] for j in range(n_vars)}

    def parents_of(struct, v):
        return struct.get(v, [])

    def has_edge(struct, u, v):
        return any(p == u for (p, _) in parents_of(struct, v))

    def get_edge_lag(struct, u, v):
        for (p, lag) in parents_of(struct, v):
            if p == u:
                return lag
        return None

    def add_edge(struct, u, v, lag):
        lag = max(1, min(int(lag), L_max))
        struct.setdefault(v, []).append((u, lag))

    def remove_edge(struct, u, v):
        if v in struct:
            struct[v] = [(p, lag) for (p, lag) in struct[v] if p != u]
            if not struct[v]:
                struct.pop(v, None)

    def set_edge_lag(struct, u, v, new_lag):
        new_lag = max(1, min(int(new_lag), L_max))
        if v in struct:
            struct[v] = [(p, (new_lag if p == u else lag)) for (p, lag) in struct[v]]

    def score(struct):
        return compute_score_mixed(data, struct, node_types, penalty_coef=penalty_coef, L_max=L_max)

    def optimize_lags_for_child(struct, child):
        improved = True
        while improved:
            improved = False
            parents = list(parents_of(struct, child))
            if not parents:
                break
            for (p, lag) in parents:
                base = score(struct)
                best_lag = lag
                best_s = base
                # +1
                if lag < L_max:
                    set_edge_lag(struct, p, child, lag + 1)
                    s_inc = score(struct)
                    if s_inc > best_s:
                        best_s = s_inc
                        best_lag = lag + 1
                # -1
                if lag > 1:
                    set_edge_lag(struct, p, child, lag - 1)
                    s_dec = score(struct)
                    if s_dec > best_s:
                        best_s = s_dec
                        best_lag = lag - 1
                set_edge_lag(struct, p, child, best_lag)
                if best_lag != lag:
                    improved = True

    # Initialize with empty DAG
    current = {j: [] for j in range(n_vars)}
    current_score = score(current)
    best_struct = copy_struct(current)
    best_score = current_score
    tabu_list = []

    # -------- Greedy hill-climb --------
    while True:
        improved = False
        best_move = None
        best_move_struct = None
        best_move_score = current_score

        for u in range(n_vars):
            for v in range(n_vars):
                # ADD u->v
                if not has_edge(current, u, v):
                    cand = copy_struct(current)
                    add_edge(cand, u, v, lag=1)
                    optimize_lags_for_child(cand, v)
                    s = score(cand)
                    if s > best_move_score:
                        best_move_score = s
                        best_move = ('add', u, v, get_edge_lag(cand, u, v), None)
                        best_move_struct = cand

                # REMOVE u->v
                if has_edge(current, u, v):
                    cand = copy_struct(current)
                    remove_edge(cand, u, v)
                    s = score(cand)
                    if s > best_move_score:
                        best_move_score = s
                        best_move = ('remove', u, v, None, get_edge_lag(current, u, v))
                        best_move_struct = cand

                # REVERSE u->v -> v->u
                # (Skip u==v: reversing a self-edge is undefined.)
                if (u != v) and has_edge(current, u, v) and not has_edge(current, v, u):
                    old_lag = get_edge_lag(current, u, v)
                    cand = copy_struct(current)
                    remove_edge(cand, u, v)
                    add_edge(cand, v, u, lag=1)
                    optimize_lags_for_child(cand, u)
                    s = score(cand)
                    if s > best_move_score:
                        new_lag = get_edge_lag(cand, v, u)
                        best_move_score = s
                        best_move = ('reverse', u, v, new_lag, old_lag)
                        best_move_struct = cand

        # LAG +/- 1
        for child, parents in list(current.items()):
            for (p, lag) in list(parents):
                if lag < L_max:
                    cand = copy_struct(current)
                    set_edge_lag(cand, p, child, lag + 1)
                    s = score(cand)
                    if s > best_move_score:
                        best_move_score = s
                        best_move = ('lag', p, child, lag + 1, lag)
                        best_move_struct = cand
                if lag > 1:
                    cand = copy_struct(current)
                    set_edge_lag(cand, p, child, lag - 1)
                    s = score(cand)
                    if s > best_move_score:
                        best_move_score = s
                        best_move = ('lag', p, child, lag - 1, lag)
                        best_move_struct = cand

        if best_move is not None and best_move_score > current_score:
            current = best_move_struct
            current_score = best_move_score
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_struct = copy_struct(current)
        if not improved:
            break

    # -------- Tabu search --------
    for _ in range(max_iter):
        best_cand_move = None
        best_cand_struct = None
        best_cand_score = -1e9

        for u in range(n_vars):
            for v in range(n_vars):
                # ADD
                if not has_edge(current, u, v):
                    cand = copy_struct(current)
                    add_edge(cand, u, v, lag=1)
                    optimize_lags_for_child(cand, v)
                    s = score(cand)
                    move = ('add', u, v, get_edge_lag(cand, u, v), None)
                    if ((move not in tabu_list) or (s > best_score)) and (s > best_cand_score):
                        best_cand_score = s
                        best_cand_move = move
                        best_cand_struct = cand

                # REMOVE
                if has_edge(current, u, v):
                    old_lag = get_edge_lag(current, u, v)
                    cand = copy_struct(current)
                    remove_edge(cand, u, v)
                    s = score(cand)
                    move = ('remove', u, v, None, old_lag)
                    if ((move not in tabu_list) or (s > best_score)) and (s > best_cand_score):
                        best_cand_score = s
                        best_cand_move = move
                        best_cand_struct = cand

                # REVERSE
                # (Skip u==v: reversing a self-edge is undefined.)
                if (u != v) and has_edge(current, u, v) and not has_edge(current, v, u):
                    old_lag = get_edge_lag(current, u, v)
                    cand = copy_struct(current)
                    remove_edge(cand, u, v)
                    add_edge(cand, v, u, lag=1)
                    optimize_lags_for_child(cand, u)
                    s = score(cand)
                    new_lag = get_edge_lag(cand, v, u)
                    move = ('reverse', u, v, new_lag, old_lag)
                    if ((move not in tabu_list) or (s > best_score)) and (s > best_cand_score):
                        best_cand_score = s
                        best_cand_move = move
                        best_cand_struct = cand

        # LAG +/- 1
        for child, parents in list(current.items()):
            for (p, lag) in list(parents):
                if lag < L_max:
                    cand = copy_struct(current)
                    set_edge_lag(cand, p, child, lag + 1)
                    s = score(cand)
                    move = ('lag', p, child, lag + 1, lag)
                    if ((move not in tabu_list) or (s > best_score)) and (s > best_cand_score):
                        best_cand_score = s
                        best_cand_move = move
                        best_cand_struct = cand
                if lag > 1:
                    cand = copy_struct(current)
                    set_edge_lag(cand, p, child, lag - 1)
                    s = score(cand)
                    move = ('lag', p, child, lag - 1, lag)
                    if ((move not in tabu_list) or (s > best_score)) and (s > best_cand_score):
                        best_cand_score = s
                        best_cand_move = move
                        best_cand_struct = cand

        if best_cand_move is None:
            break

        # Apply best admissible move
        current = best_cand_struct
        current_score = best_cand_score

        # Update Tabu list with inverse
        if len(tabu_list) >= tabu_length:
            tabu_list.pop(0)
        act, u, v, newLag, oldLag = best_cand_move
        if act == 'add':
            inverse = ('remove', u, v, None, newLag)
        elif act == 'remove':
            inverse = ('add', u, v, 1, None)
        elif act == 'reverse':
            inverse = ('reverse', v, u, oldLag if oldLag is not None else 1, newLag)
        elif act == 'lag':
            inverse = ('lag', u, v, oldLag, newLag)
        else:
            inverse = None
        if inverse is not None:
            tabu_list.append(inverse)

        # Aspiration/global best
        if current_score > best_score:
            best_score = current_score
            best_struct = copy_struct(current)

    return best_struct, best_score

