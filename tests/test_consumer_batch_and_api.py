"""
Unit tests of the streaming consumer's micro-batch logic (no Kafka/Spark needed) and of the API.

The Spark query hands each micro-batch to BatchProcessor.process as a pandas frame with
key/value/kafka_partition/kafka_offset/kafka_timestamp/event_id; we build such frames directly.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_contracts_scoring import elliptic_snapshot, make_elliptic_bundle, zenodo_rows  # noqa: E402
from src.anomaly.zenodo_detector import ZenodoAnomalyDetector  # noqa: E402
from src.anomaly.zenodo_features import compute_features  # noqa: E402
from src.contracts.adapters import elliptic_snapshot_events, zenodo_row_to_event  # noqa: E402
from src.scoring.service import ScoringConfig, ScoringService  # noqa: E402
from src.streaming_replay.consumer import BatchProcessor  # noqa: E402


@pytest.fixture(scope="module")
def zmodel(tmp_path_factory):
    d = tmp_path_factory.mktemp("z") / "1.0.0"
    det = ZenodoAnomalyDetector(n_estimators=20, max_samples=64).fit(compute_features(zenodo_rows(1500, 2)).to_numpy())
    det.save(str(d), "1.0.0")
    return str(d)


def frame(values, start_offset=0):
    rows = []
    for i, v in enumerate(values):
        try:
            eid = json.loads(v).get("event_id")
        except ValueError:
            eid = None
        rows.append({"key": eid, "value": v, "kafka_partition": 0, "kafka_offset": start_offset + i,
                     "kafka_timestamp": pd.Timestamp.now(tz="UTC"), "event_id": eid})
    return pd.DataFrame(rows)


def preds(out):
    return pd.read_parquet(os.path.join(out, "predictions"))


def test_zenodo_batch_dlq_and_sink_dedup(tmp_path, zmodel):
    svc = ScoringService(ScoringConfig(elliptic_bundle_dir=None, zenodo_model_dir=zmodel), load_elliptic=False)
    out = str(tmp_path / "run")
    bp = BatchProcessor(svc, out)
    evs = [json.dumps(zenodo_row_to_event(r)) for r in zenodo_rows(30).to_dict("records")]
    m0 = bp.process(frame(evs[:20] + ["{broken", json.dumps({"event_id": "x", "source_dataset": "zenodo"})]), 0)
    assert m0["dead_lettered"] == 2 and m0["results_written"] == 20
    # batch 1 re-delivers 5 already-scored events (escaped the watermark) + 10 new
    m1 = bp.process(frame(evs[15:30], 100), 1)
    assert m1["duplicates_dropped_at_sink"] == 5 and m1["results_written"] == 10
    p = preds(out)
    assert len(p) == 30 and p.event_id.is_unique
    assert len(pd.read_parquet(os.path.join(out, "dead_letter"))) == 2
    # replaying the same batch id (crash before checkpoint commit) overwrites, never duplicates
    bp2 = BatchProcessor(svc, out)  # fresh process: sink index is reloaded from disk
    bp2.process(frame(evs[15:30], 100), 1)
    assert len(preds(out)) == 30
    lines = open(os.path.join(out, "monitoring", "batches.jsonl")).read().splitlines()
    assert len(lines) == 3 and all("e2e_latency_ms_p50" in json.loads(x) for x in lines)


def test_elliptic_graph_snapshot_buffering_and_expiry(tmp_path, zmodel):
    bundle = make_elliptic_bundle(tmp_path / "b", "hgb_165_nbr")  # graph-dependent model
    svc = ScoringService(ScoringConfig(elliptic_bundle_dir=bundle, zenodo_model_dir=zmodel))
    out = str(tmp_path / "run")
    bp = BatchProcessor(svc, out, snapshot_ttl_batches=2)
    assert bp.graph_mode
    nodes, edges = elliptic_snapshot(n=6, step=35)
    s35 = [json.dumps(e) for e in elliptic_snapshot_events(nodes, edges)]
    nodes36, edges36 = elliptic_snapshot(n=4, step=36)
    nodes36["tx_id"] += 100
    s36 = [json.dumps(e) for e in elliptic_snapshot_events(nodes36, edges36.iloc[:0])]
    # snapshot 35 split over two batches; snapshot 36 never completes
    m0 = bp.process(frame(s35[:3] + s36[:2]), 0)
    assert m0["buffered_for_graph_context"] == 5 and m0["results_written"] == 0
    m1 = bp.process(frame(s35[3:] + s35[:1], 10), 1)  # includes a duplicate of an already-buffered event
    p = preds(out)
    assert len(p) == 6 and set(p.status) == {"scored"} and set(p.graph_context) == {"complete"}
    assert m1["results_written"] == 6
    bp.process(frame([], 20), 2)
    p = preds(out)
    rej = p[p.status == "rejected_incomplete_graph_context"]
    assert len(rej) == 2 and rej.error.str.contains("expired").all()
    assert not os.listdir(os.path.join(out, "state", "elliptic_snapshots"))
    # a late duplicate of an already-scored event must not open a new (never-completing) snapshot
    m3 = bp.process(frame(s35[:2], 30), 3)
    assert m3["duplicates_dropped_at_sink"] == 2 and m3["buffered_for_graph_context"] == 0
    bp.process(frame([], 40), 4), bp.process(frame([], 50), 5)
    assert len(preds(out)) == 8 and not os.listdir(os.path.join(out, "state", "elliptic_snapshots"))


def test_api_reads_persisted_outputs(tmp_path, zmodel):
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    svc = ScoringService(ScoringConfig(elliptic_bundle_dir=None, zenodo_model_dir=zmodel), load_elliptic=False)
    out = str(tmp_path / "run")
    bp = BatchProcessor(svc, out)
    evs = [json.dumps(zenodo_row_to_event(r)) for r in zenodo_rows(12).to_dict("records")]
    bp.process(frame(evs + ["nope"]), 0)
    c = TestClient(create_app(run_dir=out))
    s = c.get("/api/summary").json()
    assert s["predictions_total"] == 12 and s["dead_letter_total"] == 1
    assert s["by_dataset"]["zenodo"]["score_type"] == "isolation_forest_anomaly_score"
    items = c.get("/api/predictions?limit=5").json()
    assert items["total"] == 12 and len(items["items"]) == 5
    d = c.get("/api/distribution").json()
    assert sum(d["zenodo"]["counts"]) == 12
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.get("/api/dead-letter").json()["total"] == 1
    assert c.get("/").status_code == 200
    assert c.get("/api/summary?run=../../etc").status_code == 400


def test_api_empty_run(tmp_path):
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    c = TestClient(create_app(run_dir=str(tmp_path / "nothing")))
    assert c.get("/api/summary").json()["predictions_total"] == 0
    assert c.get("/api/predictions").json() == {"items": [], "total": 0}
    assert c.get("/api/health").json()["status"] == "no_data"
