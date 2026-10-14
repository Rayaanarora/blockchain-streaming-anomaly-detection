"""
Infrastructure integration test: replay producer -> Kafka -> Spark Structured Streaming -> scoring -> Parquet.

Requires the local broker:  .venv\\Scripts\\python.exe scripts\\local_kafka.py start
Run:                         .venv\\Scripts\\python.exe -m pytest tests\\integration -m integration -v
Each test run uses its own topic, so runs never interfere.

Covers: delivery of every event, malformed -> dead letter, injected duplicates dropped,
restart from checkpoint (no re-processing, only new offsets), hard kill mid-run and recovery
(no lost and no duplicated predictions), persisted outputs readable by the dashboard API.
"""
import json
import os
import subprocess
import sys
import time
import uuid

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
PY = sys.executable
BOOT = os.environ.get("BSAD_KAFKA_BOOTSTRAP", "127.0.0.1:29092")
pytestmark = pytest.mark.integration


def kafka_up():
    try:
        from confluent_kafka.admin import AdminClient
        AdminClient({"bootstrap.servers": BOOT}).list_topics(timeout=3)
        return True
    except Exception:
        return False


if not kafka_up():
    pytest.skip(f"Kafka not reachable at {BOOT} (start it with scripts/local_kafka.py start)", allow_module_level=True)


@pytest.fixture(scope="module")
def topic():
    from confluent_kafka.admin import AdminClient, NewTopic
    name = f"bsad.test.{uuid.uuid4().hex[:8]}"
    admin = AdminClient({"bootstrap.servers": BOOT})
    admin.create_topics([NewTopic(name, 3, 1)])[name].result()
    yield name  # never deleted: topic deletion crashes Kafka on Windows (delete.topic.enable=false)


def produce(topic, offset, limit, dups=0, malformed=0, manifest=None):
    cmd = [PY, os.path.join(ROOT, "scripts/replay_producer.py"), "zenodo", "--topic", topic, "--offset", str(offset),
           "--limit", str(limit), "--inject-duplicates", str(dups), "--inject-malformed", str(malformed),
           "--manifest", manifest]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    return json.load(open(manifest))


def consume(topic, run_dir, trigger="availableNow", extra=()):
    cmd = [PY, "-W", "ignore", os.path.join(ROOT, "scripts/streaming_consumer.py"), "--topic", topic,
           "--run-dir", run_dir, "--trigger", trigger, "--no-elliptic", "--max-offsets-per-trigger", "120", *extra]
    return cmd


def preds(run_dir):
    return pd.read_parquet(os.path.join(run_dir, "predictions"))


def test_end_to_end_replay_restart_and_recovery(topic, tmp_path):
    run_dir = str(tmp_path / "run")
    # 1. first replay: 300 events + 10 exact duplicates + 4 malformed
    m1 = produce(topic, 0, 300, dups=10, malformed=4, manifest=str(tmp_path / "p1.json"))
    assert m1["messages_delivered"] == 314
    r = subprocess.run(consume(topic, run_dir), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    p = preds(run_dir)
    assert p.event_id.nunique() == 300 and len(p) == 300, "every event scored exactly once"
    assert set(p.status) <= {"scored", "skipped_coinbase"}
    assert (p.model_id == "zenodo-isolation-forest").all() and (p.score_type == "isolation_forest_anomaly_score").all()
    dlq = pd.read_parquet(os.path.join(run_dir, "dead_letter"))
    assert len(dlq) == 4
    progress = [json.loads(line) for line in open(os.path.join(run_dir, "monitoring", "progress.jsonl"))]
    dropped = sum(x.get("dropped_duplicates_streaming") or 0 for x in progress)
    batches = [json.loads(line) for line in open(os.path.join(run_dir, "monitoring", "batches.jsonl"))]
    sink_dups = sum(b["duplicates_dropped_at_sink"] for b in batches)
    assert dropped + sink_dups == 10, (dropped, sink_dups)
    assert p.e2e_latency_ms.notna().all() and (p.e2e_latency_ms > 0).all()

    # 2. restart with the same checkpoint: only the 200 new events are processed
    produce(topic, 300, 200, manifest=str(tmp_path / "p2.json"))
    n_files_before = len(os.listdir(os.path.join(run_dir, "predictions")))
    r = subprocess.run(consume(topic, run_dir), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    p = preds(run_dir)
    assert p.event_id.nunique() == 500 and len(p) == 500
    assert len(os.listdir(os.path.join(run_dir, "predictions"))) > n_files_before

    # 3. hard kill mid-run, then recover
    produce(topic, 500, 600, manifest=str(tmp_path / "p3.json"))
    proc = subprocess.Popen(consume(topic, run_dir, trigger="3 seconds"), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    pred_dir = os.path.join(run_dir, "predictions")
    deadline = time.time() + 300
    start_files = len(os.listdir(pred_dir))
    while time.time() < deadline and len(os.listdir(pred_dir)) < start_files + 1:
        time.sleep(0.5)
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    proc.wait(timeout=60)
    partial = preds(run_dir)
    assert 500 < partial.event_id.nunique() < 1100, "kill should land mid-replay"
    r = subprocess.run(consume(topic, run_dir), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    p = preds(run_dir)
    assert p.event_id.nunique() == 1100, "no event lost after crash recovery"
    assert len(p) == 1100, "no duplicate prediction rows after crash recovery"
    assert list(p.sort_values(["kafka_partition", "kafka_offset"]).groupby("kafka_partition").kafka_offset
                .apply(lambda s: s.is_monotonic_increasing)) == [True] * p.kafka_partition.nunique()

    # 4. dashboard API reads the persisted outputs
    from fastapi.testclient import TestClient
    os.environ["BSAD_STREAM_RUN_DIR"] = run_dir
    from src.api.app import create_app
    c = TestClient(create_app(run_dir=run_dir))
    s = c.get("/api/summary").json()
    assert s["predictions_total"] == 1100 and s["dead_letter_total"] == 4
    assert c.get("/api/predictions?limit=5").json()["items"][0]["event_id"].startswith("zenodo:")


def _elliptic_bundle_available():
    from src.scoring.service import ScoringConfig
    try:
        return ScoringConfig().elliptic_bundle_dir is not None
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _elliptic_bundle_available(), reason="no exported Elliptic bundle under artifacts/models/elliptic")
def test_mixed_elliptic_zenodo_routing_and_graph_snapshot_parity(tmp_path):
    """Both datasets on one topic: explicit routing, complete-snapshot graph scoring, batch/stream parity."""
    from confluent_kafka.admin import AdminClient, NewTopic
    from src.models.elliptic_bundle import EllipticModelBundle
    from src.scoring.service import ScoringConfig
    name = f"bsad.test.{uuid.uuid4().hex[:8]}"
    admin = AdminClient({"bootstrap.servers": BOOT})  # keep a reference until the future resolves
    admin.create_topics([NewTopic(name, 3, 1)])[name].result()
    steps = ["48", "49"]
    subprocess.run([PY, os.path.join(ROOT, "scripts/replay_producer.py"), "elliptic", "--topic", name, "--steps",
                    *steps, "--manifest", str(tmp_path / "pe.json")], check=True, capture_output=True, timeout=600)
    produce(name, 0, 150, manifest=str(tmp_path / "pz.json"))
    n_ell = json.load(open(tmp_path / "pe.json"))["messages_delivered"]
    run_dir = str(tmp_path / "run")
    # small micro-batches so each snapshot is spread over several batches and must be buffered
    cmd = [PY, "-W", "ignore", os.path.join(ROOT, "scripts/streaming_consumer.py"), "--topic", name,
           "--run-dir", run_dir, "--trigger", "availableNow", "--max-offsets-per-trigger", "400"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:]
    p = preds(run_dir)
    assert p.event_id.is_unique and len(p) == n_ell + 150
    z, e = p[p.source_dataset == "zenodo"], p[p.source_dataset == "elliptic"]
    assert len(z) == 150 and (z.score_type == "isolation_forest_anomaly_score").all()
    assert len(e) == n_ell and (e.status == "scored").all()
    assert (e.model_id == "elliptic-supervised").all() and (e.score_type == "illicit_probability").all()
    bundle = EllipticModelBundle(ScoringConfig().elliptic_bundle_dir)
    assert set(e.graph_context) == ({"complete"} if bundle.requires_graph else {"not_required"})
    batches = [json.loads(line) for line in open(os.path.join(run_dir, "monitoring", "batches.jsonl"))]
    if bundle.requires_graph:
        assert sum(b["buffered_for_graph_context"] for b in batches) == n_ell
    # parity: streamed scores == direct batch inference on the same complete snapshots
    from src.models.elliptic_training import load_spark_output
    import numpy as np
    d = load_spark_output(os.path.join(ROOT, "data/processed/elliptic_spark"))
    for s in map(int, steps):
        m = d.time_steps == s
        nodes = pd.DataFrame(d.X_raw[m], columns=[f"f{i}" for i in range(1, 166)])
        nodes.insert(0, "tx_id", d.tx_ids[m])
        idx = np.flatnonzero(m)
        keep = np.isin(d.edge_index[0], idx)
        edges = pd.DataFrame({"source_tx_id": d.tx_ids[d.edge_index[0, keep]],
                              "target_tx_id": d.tx_ids[d.edge_index[1, keep]]})
        direct = bundle.predict(nodes, edges if bundle.requires_graph else None).set_index("tx_id")["score"]
        streamed = e[e.event_time_value == s].set_index(e[e.event_time_value == s].transaction_id.astype("int64"))["score"]
        assert len(streamed) == m.sum()
        np.testing.assert_allclose(streamed.sort_index().to_numpy(), direct.sort_index().to_numpy(), rtol=0, atol=1e-6)
