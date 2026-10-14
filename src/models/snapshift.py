"""
SnapShift-Boost: a snapshot-aware, two-stage gradient-boosting model for the Elliptic benchmark.

Motivation: Elliptic's illicit class changes after time step 43 (VERIFIED_RESULTS §1.4); a model on
raw feature values learns where illicit transactions *were* in feature space. SnapShift-Boost adds
three signals that are computed per time-step snapshot, so they are relative to the population the
transaction actually appears in:

  R  snapshot-relative features   percentile rank of each feature within the transaction's own step
  I  snapshot anomaly score       Isolation Forest fitted on the step itself (unsupervised)
  P  score propagation            stage-1 illicit probabilities of graph neighbours (stage 2 input);
                                  training rows use out-of-fold stage-1 predictions (time-grouped
                                  cross-fitting) so stage 2 never sees a probability fitted on its
                                  own label

Every function here is a pure function of one snapshot (or of independent snapshots) and is node-order
invariant, so the streaming consumer, which scores complete snapshots in arbitrary arrival order,
produces the same scores as batch scoring (see DECISIONS D-18).

Labels are used only through the fitted classifiers; nothing in R, I or P reads a label.
"""
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from src.models.elliptic_training import neighbor_aggregate_features

HGB_PARAMS = dict(learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, max_iter=200, early_stopping=False)


# ----------------------------------------------------------------------------- R: snapshot ranks
def snapshot_rank_features(X: np.ndarray, steps: np.ndarray, cols: Optional[Iterable[int]] = None) -> np.ndarray:
    """Percentile rank (average ties, in (0, 1]) of each feature within its time step."""
    cols = list(range(X.shape[1])) if cols is None else list(cols)
    out = np.empty((len(X), len(cols)), dtype=np.float32)
    for s in np.unique(steps):
        m = steps == s
        out[m] = pd.DataFrame(X[m][:, cols]).rank(method="average", pct=True).to_numpy(np.float32)
    return out


# ----------------------------------------------------------------------------- I: snapshot anomaly
def snapshot_anomaly_score(X: np.ndarray, steps: np.ndarray, tx_ids: np.ndarray, n_estimators: int = 100,
                           seed: int = 0) -> np.ndarray:
    """Isolation Forest anomaly score fitted on each step's own transactions (higher = more unusual).

    Rows are sorted by tx_id before fitting so the forest's subsamples do not depend on arrival order.
    """
    out = np.empty(len(X), dtype=np.float32)
    for s in np.unique(steps):
        idx = np.flatnonzero(steps == s)
        idx = idx[np.argsort(tx_ids[idx], kind="stable")]
        f = IsolationForest(n_estimators=n_estimators, max_samples=min(256, len(idx)), random_state=seed).fit(X[idx])
        out[idx] = -f.score_samples(X[idx])
    return out


# ----------------------------------------------------------------------------- P: score propagation
def propagation_features(p: np.ndarray, edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """Neighbour statistics of stage-1 probabilities: for in-, out- and undirected neighbours the mean and
    max, plus the undirected 2-hop mean and the node's own stage-1 probability. Order invariant
    (float64 sums, max)."""
    s, t = edge_index
    p = np.asarray(p, dtype=np.float64)
    A = sp.csr_matrix((np.ones(len(s)), (t, s)), shape=(n_nodes, n_nodes))  # row = receiver: in-neighbours
    A.data[:] = 1.0
    U = (A + A.T).tocsr()
    U.data[:] = 1.0
    cols = [p]
    for M in (A, A.T.tocsr(), U):
        deg = np.asarray(M.sum(axis=1)).ravel()
        mean = np.divide(M @ p, deg, out=np.zeros(n_nodes), where=deg > 0)
        mx = np.zeros(n_nodes)
        if M.nnz:
            M = M.tocsr()
            vals = p[M.indices]
            nz = np.diff(M.indptr) > 0
            mx[nz] = np.maximum.reduceat(vals, M.indptr[:-1][nz])
        cols += [mean, mx]
    degU = np.asarray(U.sum(axis=1)).ravel()
    m1 = np.divide(U @ p, degU, out=np.zeros(n_nodes), where=degU > 0)
    two_hop = np.divide(U @ m1, degU, out=np.zeros(n_nodes), where=degU > 0)
    cols.append(two_hop)
    return np.column_stack(cols).astype(np.float32)


# ----------------------------------------------------------------------------- feature assembly
def base_features(Xs: np.ndarray, X_raw: np.ndarray, steps: np.ndarray, tx_ids: np.ndarray,
                  edge_index: np.ndarray, components: str) -> np.ndarray:
    """Stage-1 feature matrix. ``Xs`` = train-scaled f1..f165. ``components`` ⊆ {"R", "I"}; the
    baseline block (165 + neighbour aggregates of f1..f93) is always present."""
    blocks = [Xs, neighbor_aggregate_features(Xs[:, :93], edge_index, len(Xs))]
    if "R" in components:
        blocks.append(snapshot_rank_features(X_raw, steps))
    if "I" in components:
        blocks.append(snapshot_anomaly_score(Xs, steps, tx_ids)[:, None])
    return np.hstack(blocks).astype(np.float32)


def hgb(seed: int, ratio: float, **overrides) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(class_weight={0: 1.0, 1: ratio}, random_state=seed,
                                          **{**HGB_PARAMS, **overrides})


def cross_fit_stage1(F: np.ndarray, y: np.ndarray, steps: np.ndarray, train_mask: np.ndarray, seed: int,
                     ratio: float, n_folds: int = 4):
    """Stage-1 probabilities for every node: out-of-fold for training rows (folds = contiguous groups of
    training time steps), the full-training-set model for every other row. Returns (p, full_model)."""
    full = hgb(seed, ratio).fit(F[train_mask], y[train_mask])
    p = full.predict_proba(F)[:, 1]
    tr_steps = np.unique(steps[train_mask])
    for fold_steps in np.array_split(tr_steps, n_folds):
        hold = train_mask & np.isin(steps, fold_steps)
        fit = train_mask & ~hold
        m = hgb(seed, ratio).fit(F[fit], y[fit])
        p[hold] = m.predict_proba(F[hold])[:, 1]
    return p, full


def fit_predict(data, Xs: np.ndarray, train_mask: np.ndarray, seed: int, components: str,
                ratio: Optional[float] = None, cache: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """Fit SnapShift-Boost (``components`` ⊆ "RIP") on ``train_mask`` and return probabilities for every node.

    Returns {"p": final probabilities, "p_stage1": stage-1 probabilities, "models": fitted models}.
    Without "P" the model is a single stage and both probabilities are equal.
    """
    y = data.y
    ratio = ratio or float((y[train_mask] == 0).sum() / (y[train_mask] == 1).sum())
    key = "".join(sorted(c for c in components if c in "RI"))
    if cache is not None and key in cache:
        F = cache[key]
    else:
        F = base_features(Xs, data.X_raw, data.time_steps, data.tx_ids, data.edge_index, key)
        if cache is not None:
            cache[key] = F
    if "P" not in components:
        m = hgb(seed, ratio).fit(F[train_mask], y[train_mask])
        p = m.predict_proba(F)[:, 1]
        return {"p": p, "p_stage1": p, "models": {"stage1": m}}
    p1, m1 = cross_fit_stage1(F, y, data.time_steps, train_mask, seed, ratio)
    F2 = np.hstack([F, propagation_features(p1, data.edge_index, len(F))])
    m2 = hgb(seed, ratio).fit(F2[train_mask], y[train_mask])
    return {"p": m2.predict_proba(F2)[:, 1], "p_stage1": p1, "models": {"stage1": m1, "stage2": m2}}
