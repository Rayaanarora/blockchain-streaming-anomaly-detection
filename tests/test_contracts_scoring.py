"""
Schemas, adapters, routing and scoring-output tests for the unified scoring interface.
Models are tiny and trained on synthetic data inside tmp_path (no large artifacts needed).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.anomaly.zenodo_detector import ZenodoAnomalyDetector  # noqa: E402
from src.anomaly.zenodo_features import MODEL_FEATURES, compute_features  # noqa: E402
from src.contracts.adapters import (elliptic_events_to_snapshot, elliptic_snapshot_events,  # noqa: E402
                                    zenodo_row_to_event)
from src.contracts.schemas import EventEnvelope, PredictionResult, event_json_schema  # noqa: E402
from src.models.elliptic_bundle import EllipticModelBundle, InputValidationError, export_bundle  # noqa: E402
from src.scoring.service import ScoringConfig, ScoringService  # noqa: E402

F165 = [f"f{i}" for i in range(1, 166)]


def zenodo_rows(n=50, seed=0):
    rng = np.random.default_rng(seed)
    nin = rng.integers(1, 5, n)
    nout = rng.integers(1, 5, n)
    tin = rng.integers(10_000, 10_000_000, n)
    fee = rng.integers(0, 5000, n)
    tout = tin - fee
    df = pd.DataFrame({
        "block_number": 485000 + np.arange(n) // 10, "tx_index": np.arange(n) % 10 + 1,
        "tx_hash": [f"{i:064x}" for i in range(1, n + 1)], "num_inputs": nin, "num_outputs": nout,
        "total_input_satoshi": tin, "total_output_satoshi": tout, "fee_satoshi": fee,
        "fee_rate": fee / (nin + nout), "is_coinbase": 0, "avg_input_val": tin / nin, "avg_output_val": tout / nout,
        "input_output_count_ratio": nin / nout, "input_output_val_ratio": tin / tout,
        "max_input_prior_tx_count": rng.integers(0, 20, n), "avg_input_prior_tx_count": rng.random(n) * 5,
        "max_input_prior_volume_satoshi": rng.integers(0, 10**9, n), "new_address_fraction": rng.random(n)})
    return df


@pytest.fixture(scope="module")
def zenodo_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("zen")
    X = compute_features(zenodo_rows(2000, seed=1)).to_numpy()
    det = ZenodoAnomalyDetector(n_estimators=20, max_samples=64).fit(X, meta={"test": True})
    det.save(str(d / "1.0.0"), "1.0.0")
    return str(d / "1.0.0")


def make_elliptic_bundle(root, config):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 165)).astype(np.float32)
    y = (X[:, 0] > 1).astype(int)
    sc = StandardScaler().fit(X)
    from src.models.elliptic_training import neighbor_aggregate_features
    feats = sc.transform(X)
    if config == "hgb_165_nbr":
        feats = np.hstack([feats, neighbor_aggregate_features(feats[:, :93], np.zeros((2, 0), np.int64), len(X))])
    name = "hgb_165_tuned" if config == "hgb_165_tuned" else "hgb_165_nbr"
    m = HistGradientBoostingClassifier(max_iter=10).fit(feats, y)
    return export_bundle(str(root), "0.0.1", config, {name: m}, sc, "bidirectional", 0.5, {"f1_illicit": 0.0},
                         {"note": "test"}, {"test": True})


def elliptic_snapshot(n=6, step=35):
    rng = np.random.default_rng(3)
    nodes = pd.DataFrame(rng.normal(size=(n, 165)), columns=F165)
    nodes.insert(0, "time_step", step)
    nodes.insert(0, "tx_id", np.arange(1000, 1000 + n))
    edges = pd.DataFrame({"source_tx_id": [1000, 1001, 1002], "target_tx_id": [1001, 1002, 1003]})
    return nodes, edges


# ---------------------------------------------------------------- schemas
def test_zenodo_event_roundtrip_and_stable_id():
    row = zenodo_rows(1).iloc[0].to_dict()
    ev = zenodo_row_to_event(row, seq=7)
    parsed = EventEnvelope.model_validate({"event": ev}).event
    assert parsed.source_dataset == "zenodo" and parsed.event_id == f"zenodo:{row['tx_hash']}"
    assert parsed.event_time.kind == "block_height" and parsed.replay_sequence == 7
    assert json.loads(json.dumps(ev)) == ev  # JSON-serialisable


def test_invalid_events_rejected():
    row = zenodo_rows(1).iloc[0].to_dict()
    ev = zenodo_row_to_event(row)
    bad_id = {**ev, "event_id": "zenodo:other"}
    bad_hash = {**ev, "tx_hash": "XYZ", "transaction_id": "XYZ", "event_id": "zenodo:XYZ"}
    extra = {**ev, "label": 1}
    wrong_payload = {**ev, "source_dataset": "elliptic", "event_id": f"elliptic:{ev['transaction_id']}"}
    for bad in (bad_id, bad_hash, extra, wrong_payload):
        with pytest.raises(Exception):
            EventEnvelope.model_validate({"event": bad})


def test_elliptic_events_carry_no_labels_and_rebuild_snapshot():
    nodes, edges = elliptic_snapshot()
    nodes["label"] = 1  # must not leak into events
    evs = list(elliptic_snapshot_events(nodes, edges))
    assert all("label" not in json.dumps(e) for e in evs)
    parsed = [EventEnvelope.model_validate({"event": e}).event for e in evs]
    assert all(p.graph.snapshot_size == len(nodes) for p in parsed)
    n2, e2 = elliptic_events_to_snapshot(parsed)
    assert np.allclose(n2[F165].to_numpy(), nodes[F165].to_numpy())
    assert sorted(map(tuple, e2.to_numpy())) == sorted(map(tuple, edges.to_numpy()))


def test_json_schema_exports():
    s = event_json_schema()
    assert "EllipticEvent" in json.dumps(s) and "ZenodoEvent" in json.dumps(s)
    assert "status" in PredictionResult.model_json_schema()["properties"]


# ---------------------------------------------------------------- routing + scoring
def test_routing_and_score_semantics(tmp_path, zenodo_model):
    ell = make_elliptic_bundle(tmp_path / "ell", "hgb_165_tuned")
    svc = ScoringService(ScoringConfig(elliptic_bundle_dir=ell, zenodo_model_dir=zenodo_model))
    nodes, edges = elliptic_snapshot()
    z_events = [zenodo_row_to_event(r) for r in zenodo_rows(5).to_dict("records")]
    e_events = list(elliptic_snapshot_events(nodes, edges))
    res = svc.score_events(z_events + e_events)
    assert len(res) == 11
    for r in res:
        assert r.status == "scored"
        if r.source_dataset == "zenodo":
            assert r.score_type == "isolation_forest_anomaly_score" and r.model_id == "zenodo-isolation-forest"
            assert 0 <= r.score_percentile <= 1 and len(r.reason_codes) == 3
            assert r.graph_context == "not_available"
        else:
            assert r.score_type == "illicit_probability" and r.model_id == "elliptic-supervised"
            assert 0 <= r.score <= 1 and r.graph_context == "not_required"
    assert set(svc.models()) == {"elliptic", "zenodo"}


def test_invalid_and_coinbase_handling(zenodo_model):
    svc = ScoringService(ScoringConfig(elliptic_bundle_dir=None, zenodo_model_dir=zenodo_model), load_elliptic=False)
    rows = zenodo_rows(3)
    rows.loc[1, "is_coinbase"] = 1
    rows.loc[2, "fee_satoshi"] = 1  # fee no longer equals inputs - outputs
    evs = [zenodo_row_to_event(r) for r in rows.to_dict("records")]
    res = svc.score_events(evs + ["{not json", {"source_dataset": "zenodo"}])
    st = [r.status for r in res]
    assert st[:2] == ["rejected_invalid_event", "rejected_invalid_event"]
    assert sorted(st[2:]) == ["scored", "skipped_coinbase", "skipped_invalid_input"]
    assert [r for r in res if r.status == "skipped_invalid_input"][0].error == "invalid:fee_mismatch"
    # elliptic event while elliptic model unavailable -> explicit error, never routed to zenodo
    nodes, edges = elliptic_snapshot()
    r = svc.score_events(list(elliptic_snapshot_events(nodes, edges))[:1])[0]
    assert r.status == "error" and "not loaded" in r.error


def test_graph_model_rejects_incomplete_snapshot(tmp_path, zenodo_model):
    ell = make_elliptic_bundle(tmp_path / "g", "hgb_165_nbr")
    b = EllipticModelBundle(ell)
    assert b.requires_graph
    nodes, edges = elliptic_snapshot()
    with pytest.raises(InputValidationError):
        b.predict(nodes)  # graph model without edges -> refuse
    p = b.predict(nodes.drop(columns="time_step"), edges[edges.target_tx_id != 1003].assign(target_tx_id=[1001, 1002]))
    assert set(p.graph_context) == {"complete"}
    p2 = b.predict(nodes.iloc[:2], edges)  # edges pointing outside the snapshot -> partial
    assert (p2.graph_context == "partial").any()
    svc = ScoringService(ScoringConfig(elliptic_bundle_dir=ell, zenodo_model_dir=zenodo_model))
    evs = list(elliptic_snapshot_events(nodes, edges))
    res = svc.score_events(evs[:-1])
    assert {r.status for r in res} == {"rejected_incomplete_graph_context"}
    res = svc.score_events(evs)
    assert {r.status for r in res} == {"scored"} and {r.graph_context for r in res} == {"complete"}


def test_bundle_input_validation_and_determinism(tmp_path):
    b = EllipticModelBundle(make_elliptic_bundle(tmp_path / "v", "hgb_165_tuned"))
    nodes, _ = elliptic_snapshot()
    a1, a2 = b.predict(nodes), b.predict(nodes)
    assert np.array_equal(a1.score.to_numpy(), a2.score.to_numpy())
    with pytest.raises(InputValidationError):
        b.predict(nodes.drop(columns="f7"))
    bad = nodes.copy()
    bad.loc[0, "f3"] = np.nan
    with pytest.raises(InputValidationError):
        b.predict(bad)
    with pytest.raises(InputValidationError):
        b.predict(pd.concat([nodes, nodes.iloc[:1]]))


def test_bundle_checksum_tamper_detected(tmp_path):
    d = make_elliptic_bundle(tmp_path / "t", "hgb_165_tuned")
    with open(os.path.join(d, "scaler.joblib"), "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="checksum"):
        EllipticModelBundle(d)


def test_zenodo_detector_persistence_roundtrip(zenodo_model):
    det = ZenodoAnomalyDetector.load(zenodo_model)
    X = compute_features(zenodo_rows(20, seed=9)).to_numpy()
    r1, r2 = det.score(X), ZenodoAnomalyDetector.load(zenodo_model).score(X)
    assert np.array_equal(r1["score"], r2["score"])
    assert X.shape[1] == len(MODEL_FEATURES)


def test_neighbor_features_and_graph_bundle_are_node_order_invariant(tmp_path):
    """Regression: float32 per-term scaling made neighbour means depend on node order, so the same
    snapshot arriving in a different order (interleaved Kafka partitions) got different scores."""
    from src.models.elliptic_training import neighbor_aggregate_features
    rng = np.random.default_rng(7)
    n = 3000
    # float32-quantised, standardised-looking values with repeats, like the real features
    X = np.round(rng.normal(size=(n, 93)), 3).astype(np.float32)
    e = rng.integers(0, n, size=(2, 9000))
    F = neighbor_aggregate_features(X, e, n)
    for seed in range(3):
        p = np.random.default_rng(seed).permutation(n)
        inv = np.argsort(p)
        assert np.array_equal(neighbor_aggregate_features(X[p], inv[e], n)[inv], F)

    bundle = EllipticModelBundle(make_elliptic_bundle(tmp_path / "b", "hgb_165_nbr"))
    nodes = pd.DataFrame(np.round(rng.normal(size=(400, 165)), 3), columns=F165)
    nodes.insert(0, "tx_id", np.arange(400))
    ee = rng.integers(0, 400, size=(2, 1200))
    edges = pd.DataFrame({"source_tx_id": ee[0], "target_tx_id": ee[1]})
    a = bundle.predict(nodes, edges).set_index("tx_id").score
    b = bundle.predict(nodes.sample(frac=1, random_state=1), edges.sample(frac=1, random_state=2))
    assert np.array_equal(b.set_index("tx_id").score.reindex(a.index).to_numpy(), a.to_numpy())
