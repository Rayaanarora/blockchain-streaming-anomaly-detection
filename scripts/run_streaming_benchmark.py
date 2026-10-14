"""
Local replay benchmark: Kafka -> Spark Structured Streaming (continuous trigger) -> scoring -> Parquet.

Starts the consumer with a processing-time trigger on a fresh topic, waits until it is running,
replays N existing Zenodo transactions at a fixed rate, waits until every event has a persisted
prediction, then stops the consumer and reports measured numbers only:
  end-to-end latency (producer send -> prediction written) p50/p95/p99/max, sustained throughput,
  micro-batch durations, process-tree peak RSS (python driver + JVM), failures.

    .venv\\Scripts\\python.exe scripts\\run_streaming_benchmark.py --events 6000 --rate 200 --name rate200
    .venv\\Scripts\\python.exe scripts\\run_streaming_benchmark.py --events 20000 --rate 0 --name unthrottled

Single machine, single broker, local[2] Spark: these numbers do NOT describe a cluster.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pandas as pd
import psutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
PY = sys.executable


def count_predictions(pred_dir):
    import glob
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(os.path.join(pred_dir, "*.parquet")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=6000)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--trigger", default="1 second")
    ap.add_argument("--max-offsets-per-trigger", type=int, default=5000)
    ap.add_argument("--name", default=None)
    ap.add_argument("--bootstrap", default=os.environ.get("BSAD_KAFKA_BOOTSTRAP", "127.0.0.1:29092"))
    ap.add_argument("--block-chunk", default="486000-486999")
    args = ap.parse_args()
    name = args.name or f"bench_{int(time.time())}"
    run_dir = os.path.join(ROOT, "data", "streaming", name)
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_dir} exists; choose another --name")
    topic = f"bsad.bench.{uuid.uuid4().hex[:8]}"
    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient({"bootstrap.servers": args.bootstrap})
    admin.create_topics([NewTopic(topic, 3, 1)])[topic].result()

    cons = subprocess.Popen([PY, "-W", "ignore", os.path.join(ROOT, "scripts/streaming_consumer.py"), "--topic", topic,
                             "--run-dir", run_dir, "--trigger", args.trigger, "--no-elliptic",
                             "--max-offsets-per-trigger", str(args.max_offsets_per_trigger)],
                            stdout=subprocess.DEVNULL, stderr=open(os.path.join(ROOT, "data", "temp", f"{name}.err"), "w"))
    peak = {"rss_mb": 0.0}
    stop = threading.Event()

    def sample_mem():
        while not stop.is_set():
            try:
                p = psutil.Process(cons.pid)
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                peak["rss_mb"] = max(peak["rss_mb"], rss / 2**20)
            except psutil.Error:
                pass
            time.sleep(0.25)

    threading.Thread(target=sample_mem, daemon=True).start()
    t_start = time.time()
    progress = os.path.join(run_dir, "monitoring", "progress.jsonl")
    while not os.path.exists(progress):  # first (empty) trigger has run -> query is live
        if cons.poll() is not None:
            raise SystemExit("consumer exited during startup")
        time.sleep(0.5)
    startup_s = time.time() - t_start

    prod_manifest = os.path.join(run_dir, "producer.json")
    t_prod = time.time()
    subprocess.run([PY, os.path.join(ROOT, "scripts/replay_producer.py"), "zenodo", "--topic", topic,
                    "--block-chunk", args.block_chunk, "--limit", str(args.events), "--rate", str(args.rate),
                    "--manifest", prod_manifest], check=True, capture_output=True)
    pred_dir = os.path.join(run_dir, "predictions")
    deadline = time.time() + 900
    while count_predictions(pred_dir) < args.events and time.time() < deadline:
        time.sleep(0.5)
    t_done = time.time()
    complete = count_predictions(pred_dir) >= args.events
    cons.terminate()
    subprocess.run(["taskkill", "/PID", str(cons.pid), "/T", "/F"], capture_output=True)
    stop.set()
    # topic is NOT deleted: topic deletion crashes Kafka on Windows (see infra/kafka/server.properties)

    p = pd.read_parquet(pred_dir)
    lat = p.e2e_latency_ms.dropna().to_numpy()
    batches = [json.loads(line) for line in open(os.path.join(run_dir, "monitoring", "batches.jsonl"))]
    prog = [json.loads(line) for line in open(progress) if "batch_id" in line]
    nonempty = [b for b in batches if b["input_rows"]]
    prod = json.load(open(prod_manifest))
    scored = pd.to_datetime(p.scored_at, utc=True)
    report = {
        "name": name, "complete": complete, "events": args.events, "requested_rate_eps": args.rate,
        "trigger": args.trigger, "spark_master": "local[2]", "kafka_partitions": 3,
        "consumer_startup_seconds": round(startup_s, 1),
        "producer": {k: prod[k] for k in ("events_sent", "elapsed_seconds", "send_rate_eps")},
        "predictions_written": int(len(p)), "unique_event_ids": int(p.event_id.nunique()),
        "status_counts": p.status.value_counts().to_dict(),
        "e2e_latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                           "p99": float(np.percentile(lat, 99)), "max": float(lat.max()), "n": int(len(lat))},
        "sustained_throughput_eps": round(len(p) / max((scored.max() - scored.min()).total_seconds(), 1e-9), 1),
        "wall_seconds_producer_start_to_last_prediction": round(t_done - t_prod, 2),
        "micro_batches_nonempty": len(nonempty),
        "foreachbatch_seconds": {"median": float(np.median([b["batch_seconds"] for b in nonempty])),
                                 "max": float(max(b["batch_seconds"] for b in nonempty))},
        "spark_trigger_duration_ms_median": float(np.median([x["duration_ms"]["triggerExecution"]
                                                             for x in prog if x.get("num_input_rows")])),
        "peak_process_tree_rss_mb": round(peak["rss_mb"], 1),
        "errors": int((p.status == "error").sum()),
        "definition": "e2e latency = scored_at - produced_at (producer send -> prediction row written, includes "
                      "Kafka, trigger wait, Spark micro-batch, validation, scoring and Parquet write)",
    }
    out = os.path.join(ROOT, "data", "validation", "streaming_replay", f"{name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
