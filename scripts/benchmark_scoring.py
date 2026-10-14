"""
In-memory scoring benchmarks for the exported models (no Kafka, no Spark).

Measures, separately:
  * model load time (bundle files + checksum verification)
  * Elliptic: graph-context construction (events -> node/edge frames) and model inference per complete
    time-step snapshot, for test steps 35-49
  * Zenodo: single-event and batch latency through the unified ScoringService (validation + features + model)
These are in-process numbers; end-to-end Kafka/Spark latency is in data/validation/streaming_replay/.

    .venv\\Scripts\\python.exe scripts\\benchmark_scoring.py
Writes data/validation/scoring_benchmark.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def pct(xs):
    a = np.asarray(xs) * 1000.0
    return {"p50_ms": round(float(np.percentile(a, 50)), 3), "p95_ms": round(float(np.percentile(a, 95)), 3),
            "max_ms": round(float(a.max()), 3), "n": int(len(a))}


def main():
    from src.contracts.adapters import elliptic_events_to_snapshot, elliptic_snapshot_events, zenodo_row_to_event
    from src.contracts.schemas import EventEnvelope
    from src.models.elliptic_bundle import EllipticModelBundle, latest_bundle_dir
    from src.anomaly.zenodo_detector import ZenodoAnomalyDetector
    from src.scoring.service import ScoringConfig, ScoringService
    from src.streaming_replay.producer import zenodo_source

    proc = psutil.Process()
    out = {"host_cpu_count": os.cpu_count(), "torch_threads": None}
    cfg = ScoringConfig()
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        EllipticModelBundle(cfg.elliptic_bundle_dir)
        t.append(time.perf_counter() - t0)
    out["elliptic_bundle"] = cfg.elliptic_bundle_dir
    out["elliptic_model_load_seconds_median"] = round(float(np.median(t)), 4)
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        ZenodoAnomalyDetector.load(cfg.zenodo_model_dir)
        t.append(time.perf_counter() - t0)
    out["zenodo_model_load_seconds_median"] = round(float(np.median(t)), 4)
    svc = ScoringService(cfg)
    b = svc.elliptic
    import torch
    out["torch_threads"] = torch.get_num_threads()
    out["elliptic_config"] = b.manifest["config"]
    out["elliptic_requires_graph"] = b.requires_graph

    # ---------------- Elliptic snapshots (test steps)
    d = os.path.join(ROOT, "data/processed/elliptic_spark")
    nodes = pd.read_parquet(os.path.join(d, "nodes"), filters=[("time_step", ">=", 35)])
    nodes["time_step"] = nodes["time_step"].astype(int)
    edges = pd.read_parquet(os.path.join(d, "edges"), filters=[("time_step", ">=", 35)])
    ctx_t, inf_t, sizes, total_t = [], [], [], []
    labels_all, scores_all = [], []
    for step, g in nodes.groupby("time_step"):
        e = edges[edges.time_step == step]
        evs = [EventEnvelope.model_validate({"event": x}).event for x in
               elliptic_snapshot_events(g.drop(columns=["label", "label_status", "split", "node_idx"]), e)]
        t0 = time.perf_counter()
        nf, ef = elliptic_events_to_snapshot(evs)
        t1 = time.perf_counter()
        pred = b.predict(nf, ef if b.requires_graph else None)
        t2 = time.perf_counter()
        ctx_t.append(t1 - t0)
        inf_t.append(t2 - t1)
        sizes.append(len(g))
        t0 = time.perf_counter()
        svc.score_events(evs)
        total_t.append(time.perf_counter() - t0)
        lab = g.set_index("tx_id").loc[pred.tx_id, "label"].to_numpy()
        labels_all.append(lab)
        scores_all.append(pred.score.to_numpy())
    y = np.concatenate(labels_all)
    s = np.concatenate(scores_all)
    m = y != -1
    from sklearn.metrics import f1_score
    out["elliptic_snapshot"] = {
        "snapshots": len(sizes), "nodes_per_snapshot_median": int(np.median(sizes)),
        "graph_context_construction": pct(ctx_t), "model_inference": pct(inf_t),
        "full_service_call_incl_validation": pct(total_t),
        "per_transaction_us_median": round(float(np.median(np.array(inf_t) / np.array(sizes)) * 1e6), 2),
        "sanity_check_streaming_path_test_f1": round(float(f1_score(y[m], (s[m] >= b.threshold).astype(int))), 4),
        "note": "sanity_check_* recomputes test F1 through the event/adapter path to prove it matches the "
                "offline evaluation of the exported model; it is not a new result."}

    # ---------------- Zenodo
    df = zenodo_source(os.path.join(ROOT, "data/parquet/zenodo_features"), "489000-489999", 5000)
    evs = [zenodo_row_to_event(r) for r in df.to_dict("records")]
    single = []
    for e in evs[:300]:
        t0 = time.perf_counter()
        svc.score_events([e])
        single.append(time.perf_counter() - t0)
    batch = {}
    for bs in (100, 1000, 5000):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            svc.score_events(evs[:bs])
            ts.append(time.perf_counter() - t0)
        batch[str(bs)] = {"median_seconds": round(float(np.median(ts)), 4),
                          "events_per_second": round(bs / float(np.median(ts)), 1)}
    out["zenodo"] = {"single_event_latency": pct(single), "batch": batch}
    out["process_rss_mb"] = round(proc.memory_info().rss / 2**20, 1)
    path = os.path.join(ROOT, "data/validation/scoring_benchmark.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
