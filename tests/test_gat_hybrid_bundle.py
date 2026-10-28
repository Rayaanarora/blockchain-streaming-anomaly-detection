"""
Tests for the deep-learning hybrid bundle (config ens_hgb_hgbgat: GAT embeddings + HistGBDT ensemble).
A tiny GAT and tree models are trained on synthetic data inside tmp_path; no large artefacts needed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.elliptic_bundle import EllipticModelBundle, InputValidationError, export_bundle  # noqa: E402
from src.models.graph_encoder import GraphAttentionEncoder, make_edge_index  # noqa: E402

F165 = [f"f{i}" for i in range(1, 166)]


def snapshot(n=40, seed=0):
    rng = np.random.default_rng(seed)
    nodes = pd.DataFrame(rng.normal(size=(n, 165)), columns=F165)
    nodes.insert(0, "tx_id", np.arange(5000, 5000 + n))
    src = rng.integers(0, n, 60)
    dst = rng.integers(0, n, 60)
    keep = src != dst
    edges = pd.DataFrame({"source_tx_id": 5000 + src[keep], "target_tx_id": 5000 + dst[keep]})
    return nodes, edges


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    torch.manual_seed(0)
    nodes, edges = snapshot(300, seed=1)
    X_raw = nodes[F165].to_numpy(np.float32)
    y = (X_raw[:, 0] + X_raw[:, 1] > 1).astype(int)
    sc = StandardScaler().fit(X_raw)
    X = sc.transform(X_raw).astype(np.float32)
    pos = {t: i for i, t in enumerate(nodes.tx_id)}
    e = np.vstack([edges.source_tx_id.map(pos).to_numpy(), edges.target_tx_id.map(pos).to_numpy()])
    gat = GraphAttentionEncoder(165, hidden_channels=8, heads=2, dropout=0.0, embedding_dim=32)
    gat.eval()
    _, emb = gat.predict(X, make_edge_index(e, "bidirectional"))
    m_tab = HistGradientBoostingClassifier(max_iter=10).fit(X, y)
    m_fused = HistGradientBoostingClassifier(max_iter=10).fit(np.hstack([X, emb]), y)
    root = tmp_path_factory.mktemp("hyb")
    return export_bundle(str(root), "9.9.9", "ens_hgb_hgbgat",
                         {"hgb_165_tuned": m_tab, "hgb_165_gat_cf": m_fused, "gat": gat}, sc, "bidirectional",
                         0.5, {"f1_illicit": 0.0}, {"note": "test"}, {"test": True}, seed=123)


def test_manifest_describes_deep_learning_hybrid(bundle_dir):
    m = json.load(open(os.path.join(bundle_dir, "manifest.json")))
    assert m["config"] == "ens_hgb_hgbgat" and m["requires_graph"] is True
    assert m["gat_edge_direction"] == "bidirectional" and "gat" in m["files"]
    assert m["export_seed"] == 123 and "test_metrics_seed123" in m and "seed 123" in m["threshold_source"]
    assert [c["name"] for c in m["components"]] == ["hgb_165_tuned", "hgb_165_gat_cf"]


def test_requires_graph_context(bundle_dir):
    b = EllipticModelBundle(bundle_dir)
    nodes, edges = snapshot()
    with pytest.raises(InputValidationError, match="graph context"):
        b.predict(nodes)
    p = b.predict(nodes, edges)
    assert set(p.graph_context) == {"complete"} and p.score.between(0, 1).all()


def test_deterministic_and_reload_identical(bundle_dir):
    nodes, edges = snapshot()
    a = EllipticModelBundle(bundle_dir).predict(nodes, edges).score.to_numpy()
    b = EllipticModelBundle(bundle_dir).predict(nodes, edges).score.to_numpy()
    assert np.array_equal(a, b)


def test_score_is_mean_of_tabular_and_gat_branch(bundle_dir):
    b = EllipticModelBundle(bundle_dir)
    nodes, edges = snapshot()
    score = b.predict(nodes, edges).score.to_numpy()
    X = b.scaler.transform(nodes[F165].to_numpy(np.float32)).astype(np.float32)
    pos = {t: i for i, t in enumerate(nodes.tx_id)}
    e = np.vstack([edges.source_tx_id.map(pos).to_numpy(), edges.target_tx_id.map(pos).to_numpy()])
    _, emb = b.gat.predict(X, make_edge_index(e, "bidirectional"))
    comps = {n: m for n, _, m in b.components}
    expect = 0.5 * (comps["hgb_165_tuned"].predict_proba(X)[:, 1]
                    + comps["hgb_165_gat_cf"].predict_proba(np.hstack([X, emb]))[:, 1])
    assert np.allclose(score, expect, atol=1e-6)


def test_graph_changes_the_score(bundle_dir):
    """The GAT branch really uses the edges: removing them changes some scores."""
    b = EllipticModelBundle(bundle_dir)
    nodes, edges = snapshot()
    with_edges = b.predict(nodes, edges).score.to_numpy()
    without = b.predict(nodes, edges.iloc[:0]).score.to_numpy()
    assert not np.allclose(with_edges, without)
