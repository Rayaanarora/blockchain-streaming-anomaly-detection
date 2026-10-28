"""Unit tests for src/evaluation (metrics, stats, preprocessing helpers, TGAT-BoostNet wiring)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef,
                             roc_auc_score)

from src.evaluation import stats as ST
from src.evaluation.data import Elliptic, induced
from src.evaluation.metrics import expected_calibration_error, full_metrics, per_timestep
from src.evaluation.preprocessing import agreement_table, class_snr, quality_profile


@pytest.fixture
def yp():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    p = np.clip(0.6 * y + rng.normal(0.2, 0.2, 2000), 0, 1)
    return y, p


def test_full_metrics_match_sklearn(yp):
    y, p = yp
    m = full_metrics(y, p, 0.5)
    pred = (p >= 0.5).astype(int)
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))
    assert m["kappa"] == pytest.approx(cohen_kappa_score(y, pred))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert m["f1_macro"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["f1_weighted"] == pytest.approx(f1_score(y, pred, average="weighted"))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == len(y)
    assert m["fpr"] == pytest.approx(1 - m["specificity"])


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(1)
    p = rng.random(200000)
    y = (rng.random(200000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_per_timestep_counts(yp):
    y, p = yp
    steps = np.repeat(np.arange(35, 45), 200)
    rows = per_timestep(y, p, steps)
    assert [r["time_step"] for r in rows] == list(range(35, 45))
    assert sum(r["n"] for r in rows) == len(y)


def test_mcnemar_and_holm():
    y = np.array([1] * 50 + [0] * 50)
    a = y.copy()
    b = y.copy()
    b[:20] = 0  # b wrong on 20 positives where a is right
    r = ST.mcnemar(y, a, b)
    assert r["a_right_b_wrong"] == 20 and r["a_wrong_b_right"] == 0
    assert r["p_value"] < 1e-4
    adj = ST.holm({"x": 0.01, "y": 0.04, "z": 0.5})
    assert adj["x"] == pytest.approx(0.03) and adj["y"] == pytest.approx(0.08) and adj["z"] == pytest.approx(0.5)


def test_paired_bootstrap_detects_difference(yp):
    y, p = yp
    worse = np.clip(p + np.random.default_rng(2).normal(0, 0.3, len(p)), 0, 1)
    f1 = lambda yy, pp: f1_score(yy, (pp >= 0.5).astype(int))
    r = ST.paired_bootstrap(y, [p], [worse], f1, n_boot=300)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["p_value"] < 0.05


def test_friedman_nemenyi_ranks():
    scores = pd.DataFrame({"a": [0.9, 0.8, 0.85, 0.7], "b": [0.5, 0.4, 0.6, 0.3], "c": [0.7, 0.6, 0.65, 0.5]})
    r = ST.friedman_nemenyi(scores)
    assert list(r["avg_ranks"].index) == ["a", "c", "b"]
    assert r["avg_ranks"]["a"] == 1.0 and r["cd"] > 0


def test_agreement_table_zero_and_nonzero_error():
    a = pd.DataFrame({"x": np.arange(10.0), "y": np.linspace(-1, 1, 10)})
    t = agreement_table(a, a.copy(), ["x", "y"])
    assert (t.RMSE == 0).all() and (t.exact_match_rate == 1).all() and (t.R2 == 1).all()
    b = a.copy()
    b.loc[0, "x"] += 1
    t2 = agreement_table(a, b, ["x"])
    assert t2.loc["x", "max_abs_err"] == 1 and t2.loc["x", "exact_match_rate"] == pytest.approx(0.9)


def test_class_snr_and_skew_invariant_to_affine():
    rng = np.random.default_rng(3)
    X = rng.lognormal(size=(500, 4))
    y = (rng.random(500) < 0.3).astype(int)
    X2 = 3.0 * X - 7.0
    assert np.allclose(class_snr(X, y), class_snr(X2, y))
    q1, q2 = quality_profile(X, list("abcd")), quality_profile(X2, list("abcd"))
    assert np.allclose(q1.skewness, q2.skewness) and np.allclose(q1.outlier_z3, q2.outlier_z3)


def test_induced_subgraph_keeps_only_internal_edges():
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    idx, e = induced(np.array([True, True, False, True]), edges)
    assert list(idx) == [0, 1, 3]
    assert sorted(map(tuple, e.T.tolist())) == [(0, 1), (2, 0)]


def _toy_elliptic(n_steps=6, per_step=60, seed=0):
    rng = np.random.default_rng(seed)
    steps = np.repeat(np.arange(1, n_steps + 1), per_step)
    n = len(steps)
    y = np.where(rng.random(n) < 0.15, 1, 0)
    y[rng.random(n) < 0.3] = -1
    X = rng.normal(size=(n, 165)).astype(np.float32)
    X[:, 0] += 2.0 * (y == 1)
    src, dst = [], []
    for s in range(1, n_steps + 1):
        ids = np.flatnonzero(steps == s)
        src += list(ids[:-1])
        dst += list(ids[1:])
    return Elliptic(np.arange(n), steps, X, y, np.array([src, dst]))


def test_tgat_boostnet_cross_fitting_and_shapes(monkeypatch):
    import src.evaluation.tgat_boostnet as T
    import src.evaluation.data as D
    monkeypatch.setattr(T, "TRAIN_STEPS", (1, 4))
    monkeypatch.setattr(T, "TEST_STEPS", (5, 6))
    monkeypatch.setattr(D, "TRAIN_STEPS", (1, 4))
    monkeypatch.setattr(D, "TEST_STEPS", (5, 6))
    data = _toy_elliptic()
    seen = []
    orig = T.train_gat

    def spy(X, edges, y, node_mask, cfg, seed, device="cpu"):
        seen.append(set(np.unique(data.time_steps[node_mask])))
        return orig(X, edges, y, node_mask, cfg, seed, device)
    monkeypatch.setattr(T, "train_gat", spy)
    cfg = T.TGATBoostNetConfig(epochs=2, n_folds=2, gbdt=dict(n_estimators=5, max_depth=2))
    r = T.fit_tgat_boostnet(data, cfg, seed=0, log=lambda *a: None)
    assert r["p_test"].shape == (data.test_mask.sum(),)
    assert r["emb_train"].shape == (data.train_mask.sum(), 32)
    assert r["n_features"] == 165 + 32
    # first GAT sees all training steps; each fold GAT never sees the steps it embeds
    assert seen[0] == {1, 2, 3, 4} and seen[1] == {3, 4} and seen[2] == {1, 2}
    assert all(5 not in s and 6 not in s for s in seen)
    assert r["infer"](5).shape == ((data.time_steps == 5).sum(),)


def test_probability_error_metrics_definitions():
    from src.evaluation.metrics import probability_error_metrics
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 1.0])
    r = probability_error_metrics(y, p)
    e = p - y
    assert r["MSE"] == pytest.approx((e ** 2).mean()) and r["RMSE"] == pytest.approx(np.sqrt((e ** 2).mean()))
    assert r["MAE"] == pytest.approx(np.abs(e).mean()) and r["ME_bias"] == pytest.approx(e.mean())
    assert r["MAPE_pct"] == pytest.approx(100 * (0.4 + 0.0) / 2)  # illicit rows only
    assert r["MedAE"] == pytest.approx(np.median(np.abs(e)))
    assert r["R2"] == pytest.approx(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


def test_agreement_table_course_metrics():
    a = pd.DataFrame({"x": [0.0, 2.0, 4.0, 10.0]})
    b = pd.DataFrame({"x": [0.0, 3.0, 4.0, 10.0]})
    t = agreement_table(a, b, ["x"]).loc["x"]
    assert t.MSE == pytest.approx(0.25) and t.RMSE == pytest.approx(0.5) and t.ME_bias == pytest.approx(0.25)
    assert t.MAPE_pct == pytest.approx(100 * (0.5 + 0 + 0) / 3)  # zero reference excluded
    assert t.NRMSE == pytest.approx(0.5 / 10) and t.MedAE == pytest.approx(0.0)
