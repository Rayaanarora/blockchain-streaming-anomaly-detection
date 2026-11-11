"""
Tests for the TGAT-BoostNet bundle (config tgat_boostnet: 8-head GAT embedding + XGBoost on [165 || 32]) and
its FixedStandardiser. Synthetic data only; a tiny GAT and a 10-tree XGBoost are fitted inside tmp_path.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.tgat_boostnet import _standardise_on  # noqa: E402
from src.models.elliptic_bundle import (EllipticModelBundle, FixedStandardiser, InputValidationError,  # noqa: E402
                                        export_bundle)
from src.models.graph_encoder import GraphAttentionEncoder, make_edge_index  # noqa: E402

F165 = [f"f{i}" for i in range(1, 166)]


def snapshot(n=40, seed=0):
    rng = np.random.default_rng(seed)
    # discrete-valued, offset and scaled like Elliptic's features, with one constant column (sd = 0)
    X = (rng.integers(-50, 50, size=(n, 165)) * 0.0371 + 3.7).astype(np.float32)
    X[:, 7] = 2.5
    nodes = pd.DataFrame(X, columns=F165)
    nodes.insert(0, "tx_id", np.arange(5000, 5000 + n))
    src = rng.integers(0, n, 60)
    dst = rng.integers(0, n, 60)
    keep = src != dst
    edges = pd.DataFrame({"source_tx_id": 5000 + src[keep], "target_tx_id": 5000 + dst[keep]})
    return nodes, edges


def stats(X, rows):
    mu = X[rows].astype(np.float64).mean(0)
    sd = X[rows].astype(np.float64).std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def test_fixed_standardiser_bit_equal_to_standardise_on():
    nodes, _ = snapshot(500, seed=3)
    X = nodes[F165].to_numpy(np.float32)
    rows = np.zeros(len(X), dtype=bool)
    rows[:350] = True
    mu, sd = stats(X, rows)
    got = FixedStandardiser(mu, sd).transform(X).astype(np.float32)
    ref = _standardise_on(X, rows)
    assert got.dtype == ref.dtype == np.float32
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
    # float64 input is reduced to float32 first, exactly like the bundle's validate_nodes
    assert np.array_equal(FixedStandardiser(mu, sd).transform(X.astype(np.float64)).astype(np.float32), ref)


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    torch.manual_seed(0)
    nodes, edges = snapshot(300, seed=1)
    X_raw = nodes[F165].to_numpy(np.float32)
    mu, sd = stats(X_raw, np.ones(len(X_raw), dtype=bool))
    sc = FixedStandardiser(mu, sd)
    X = sc.transform(X_raw).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0.5).astype(int)
    pos = {t: i for i, t in enumerate(nodes.tx_id)}
    e = np.vstack([edges.source_tx_id.map(pos).to_numpy(), edges.target_tx_id.map(pos).to_numpy()])
    gat = GraphAttentionEncoder(165, hidden_channels=4, heads=8, dropout=0.3, embedding_dim=32)
    gat.eval()
    _, emb = gat.predict(X, make_edge_index(e, "bidirectional"))
    booster = xgb.XGBClassifier(n_estimators=10, max_depth=3, tree_method="hist", device="cpu", n_jobs=1,
                                random_state=0).fit(np.hstack([X, emb]), y)
    root = tmp_path_factory.mktemp("tgat")
    return export_bundle(str(root), "9.9.9", "tgat_boostnet", {"tgat_boostnet_xgb": booster, "gat": gat}, sc,
                         "bidirectional", 0.37, {"f1_illicit": 0.0}, {"note": "test"}, {"test": True}, seed=42,
                         score_semantics="test semantics", threshold_source="test threshold source")


def test_manifest(bundle_dir):
    m = json.load(open(os.path.join(bundle_dir, "manifest.json")))
    assert m["config"] == "tgat_boostnet" and m["requires_graph"] is True
    assert m["gat_edge_direction"] == "bidirectional" and m["files"]["gat"] == "gat.pt"
    assert [c["name"] for c in m["components"]] == ["tgat_boostnet_xgb"]
    assert m["components"][0]["file"].endswith(".ubj") and m["components"][0]["features"] == "x165_emb"
    assert m["decision_threshold"] == 0.37 and m["export_seed"] == 42
    assert m["score_semantics"] == "test semantics" and m["threshold_source"] == "test threshold source"
    assert os.path.exists(os.path.join(bundle_dir, "gat.pt"))


def test_predict_equals_manual_pipeline(bundle_dir):
    b = EllipticModelBundle(bundle_dir)
    assert isinstance(b.scaler, FixedStandardiser)
    assert b.gat.config["heads"] == 8 and b.gat.config["embedding_dim"] == 32
    nodes, edges = snapshot(seed=5)
    score = b.predict(nodes, edges).score.to_numpy()
    X = b.scaler.transform(nodes[F165].to_numpy(np.float32)).astype(np.float32)
    pos = {t: i for i, t in enumerate(nodes.tx_id)}
    e = np.vstack([edges.source_tx_id.map(pos).to_numpy(), edges.target_tx_id.map(pos).to_numpy()])
    _, emb = b.gat.predict(X, make_edge_index(e, "bidirectional"))
    (_, _, booster), = b.components
    expect = booster.predict_proba(np.hstack([X, emb]))[:, 1]
    assert np.array_equal(score, expect.astype(float))


def test_requires_graph_context(bundle_dir):
    b = EllipticModelBundle(bundle_dir)
    nodes, edges = snapshot()
    with pytest.raises(InputValidationError, match="graph context"):
        b.predict(nodes)
    p = b.predict(nodes, edges)
    assert set(p.graph_context) == {"complete"} and p.score.between(0, 1).all()
