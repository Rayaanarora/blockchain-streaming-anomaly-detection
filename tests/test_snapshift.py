"""
SnapShift-Boost: feature correctness, leakage guards, node-order invariance and the two-stage bundle.
Synthetic data only (no large artifacts needed).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import snapshift  # noqa: E402
from src.models.elliptic_bundle import EllipticModelBundle, InputValidationError, export_bundle  # noqa: E402
from src.models.elliptic_training import EllipticData  # noqa: E402

F165 = [f"f{i}" for i in range(1, 166)]
FAST = dict(max_iter=15)


@pytest.fixture(autouse=True)
def small_trees(monkeypatch):
    monkeypatch.setattr(snapshift, "HGB_PARAMS", {**snapshift.HGB_PARAMS, **FAST})


def synthetic(n_steps=8, per_step=120, seed=0):
    """Steps with a planted signal: illicit nodes have high f1 and link to each other."""
    rng = np.random.default_rng(seed)
    n = n_steps * per_step
    steps = np.repeat(np.arange(1, n_steps + 1), per_step)
    y = (rng.random(n) < 0.15).astype(int)
    X = rng.normal(size=(n, 165)).astype(np.float32)
    X[:, 0] += 1.5 * y
    edges = []
    for s in range(1, n_steps + 1):
        idx = np.flatnonzero(steps == s)
        bad = idx[y[idx] == 1]
        edges.append(rng.choice(idx, size=(2, per_step * 2)))
        if len(bad) > 1:
            edges.append(rng.choice(bad, size=(2, len(bad) * 2)))
    e = np.hstack(edges)
    e = e[:, e[0] != e[1]]
    y_obs = y.copy()
    y_obs[rng.random(n) < 0.3] = -1  # unknown labels
    return EllipticData(np.arange(n) * 3 + 11, steps, X, y_obs, e.astype(np.int64)), y


def test_propagation_features_match_brute_force_and_are_order_invariant():
    rng = np.random.default_rng(1)
    n = 60
    e = rng.integers(0, n, (2, 150))
    e = e[:, e[0] != e[1]]
    p = rng.random(n)
    F = snapshift.propagation_features(p, e, n)
    for i in range(n):
        ins, outs = e[0][e[1] == i], e[1][e[0] == i]
        und = np.unique(np.concatenate([ins, outs]))
        ins, outs = np.unique(ins), np.unique(outs)
        exp = [p[i]]
        for nb in (ins, outs, und):
            exp += [p[nb].mean() if len(nb) else 0.0, p[nb].max() if len(nb) else 0.0]
        assert np.allclose(F[i, :7], exp, atol=1e-6)
    perm = rng.permutation(n)
    inv = np.argsort(perm)
    assert np.array_equal(snapshift.propagation_features(p[perm], inv[e], n), F[perm])


def test_stage1_training_scores_are_out_of_fold():
    data, _ = synthetic()
    tr = (data.y >= 0) & (data.time_steps <= 6)
    F = snapshift.base_features(StandardScaler().fit(data.X_raw[tr]).transform(data.X_raw).astype(np.float32),
                                data.X_raw, data.time_steps, data.tx_ids, data.edge_index, "")
    ratio = float((data.y[tr] == 0).sum() / (data.y[tr] == 1).sum())
    p, full = snapshift.cross_fit_stage1(F, data.y, data.time_steps, tr, seed=0, ratio=ratio, n_folds=3)
    hold = tr & np.isin(data.time_steps, [1, 2])  # first fold of training steps 1..6
    m = snapshift.hgb(0, ratio).fit(F[tr & ~hold], data.y[tr & ~hold])
    assert np.allclose(p[hold], m.predict_proba(F[hold])[:, 1])
    assert not np.allclose(p[hold], full.predict_proba(F[hold])[:, 1])  # not the in-sample model
    rest = ~tr
    assert np.allclose(p[rest], full.predict_proba(F[rest])[:, 1])


def test_labels_outside_training_rows_are_never_read():
    data, _ = synthetic()
    tr = (data.y >= 0) & (data.time_steps <= 6)
    Xs = StandardScaler().fit(data.X_raw[tr]).transform(data.X_raw).astype(np.float32)
    a = snapshift.fit_predict(data, Xs, tr, seed=0, components="P")["p"]
    y2 = data.y.copy()
    y2[~tr] = np.where(y2[~tr] == -1, 1, 1 - np.maximum(y2[~tr], 0))  # scramble every non-training label
    d2 = EllipticData(data.tx_ids, data.time_steps, data.X_raw, y2, data.edge_index)
    b = snapshift.fit_predict(d2, Xs, tr, seed=0, components="P")["p"]
    assert np.array_equal(a, b)


def make_two_stage_bundle(root):
    data, _ = synthetic()
    tr = (data.y >= 0) & (data.time_steps <= 6)
    sc = StandardScaler().fit(data.X_raw[tr])
    out = snapshift.fit_predict(data, sc.transform(data.X_raw).astype(np.float32), tr, seed=0, components="P")
    path = export_bundle(str(root), "0.0.1", "snapshift_P",
                         {"snapshift_stage1": out["models"]["stage1"], "snapshift_stage2": out["models"]["stage2"]},
                         sc, None, 0.5, {"f1_illicit": 0.0}, {"note": "test"}, {"test": True})
    return path, data, out


def test_two_stage_bundle_reproduces_fit_per_snapshot_and_is_order_invariant(tmp_path):
    path, data, out = make_two_stage_bundle(tmp_path)
    b = EllipticModelBundle(path)
    assert b.requires_graph and b.manifest["combine"] == "two_stage"
    s = 8
    idx = np.flatnonzero(data.time_steps == s)
    nodes = pd.DataFrame(data.X_raw[idx], columns=F165)
    nodes.insert(0, "tx_id", data.tx_ids[idx])
    em = data.time_steps[data.edge_index[0]] == s
    edges = pd.DataFrame({"source_tx_id": data.tx_ids[data.edge_index[0][em]],
                          "target_tx_id": data.tx_ids[data.edge_index[1][em]]})
    res = b.predict(nodes, edges)
    assert np.allclose(res.score.to_numpy(), out["p"][idx], atol=1e-12)  # snapshot scoring = full-graph fit
    shuffled = b.predict(nodes.sample(frac=1, random_state=3), edges.sample(frac=1, random_state=4))
    assert np.array_equal(shuffled.set_index("tx_id").score.reindex(res.tx_id).to_numpy(), res.score.to_numpy())
    with pytest.raises(InputValidationError):
        b.predict(nodes)  # graph context is mandatory
