"""
Spark Structured Streaming consumer: Kafka -> parse/validate -> de-duplicate -> score -> durable outputs.

Pipeline (one streaming query):
  Kafka source (offsets tracked in the checkpoint)
  -> JVM-side parse: from_json of the envelope fields (event_id, source_dataset, schema_version)
  -> dedup_key = event_id, or "malformed:<partition>:<offset>" for unparseable messages
  -> withWatermark(kafka timestamp, dedup_watermark) + dropDuplicatesWithinWatermark(dedup_key)
     (state kept in the checkpoint's state store, expired after the watermark: duplicates that
      arrive later than the watermark are NOT detected by Spark and are counted by the sink)
  -> foreachBatch (runs in the driver):
       * full contract validation + routing via ScoringService
       * Elliptic + graph-dependent model: events are buffered per time-step snapshot in a durable
         driver-side state directory until the snapshot is complete (or expired -> explicit
         rejected_incomplete_graph_context results); tabular models score immediately
       * predictions -> <out>/predictions/batch=<id>.parquet  (atomic replace, so a replayed batch id
         overwrites instead of duplicating)
       * malformed / invalid events -> <out>/dead_letter/batch=<id>.parquet
       * per-batch metrics -> <out>/monitoring/batches.jsonl
  StreamingQueryListener -> <out>/monitoring/progress.jsonl (input rows, durations, dropped duplicates)

Python UDFs are avoided on purpose: Spark Python workers crash on this host (see DECISIONS.md D-05),
so parsing/dedup run in the JVM and model scoring runs in the driver inside foreachBatch.
Delivery semantics: Kafka offsets + checkpoint give at-least-once batch execution; per-batch file
names make the file sink idempotent for replayed batches; sink-side event_id dedup removes
duplicates the watermark missed. This is NOT verified end-to-end exactly-once.
"""
import glob
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.contracts.schemas import EventEnvelope, PredictionResult, utc_now_iso
from src.scoring.service import ScoringService

log = logging.getLogger("bsad.consumer")

PRED_SCHEMA = pa.schema([
    ("event_id", pa.string()), ("transaction_id", pa.string()), ("source_dataset", pa.string()),
    ("event_time_kind", pa.string()), ("event_time_value", pa.int64()), ("model_id", pa.string()),
    ("model_version", pa.string()), ("score", pa.float64()), ("score_type", pa.string()),
    ("score_percentile", pa.float64()), ("threshold", pa.float64()), ("flagged", pa.bool_()),
    ("reason_codes", pa.string()), ("graph_context", pa.string()), ("status", pa.string()), ("error", pa.string()),
    ("produced_at", pa.string()), ("kafka_timestamp", pa.string()), ("processing_time", pa.string()),
    ("scored_at", pa.string()), ("kafka_partition", pa.int32()), ("kafka_offset", pa.int64()),
    ("batch_id", pa.int64()), ("e2e_latency_ms", pa.float64()),
])
DLQ_SCHEMA = pa.schema([("kafka_partition", pa.int32()), ("kafka_offset", pa.int64()), ("key", pa.string()),
                        ("value", pa.string()), ("reason", pa.string()), ("batch_id", pa.int64()),
                        ("detected_at", pa.string())])


def _iso_to_epoch_ms(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000.0


def _atomic_parquet(table: pa.Table, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, path)


class SnapshotBuffer:
    """Durable per-snapshot buffer for graph-dependent Elliptic scoring (driver-side state).

    Files: <state_dir>/<snapshot>.jsonl (one event JSON per line, appended per batch) and
    <snapshot>.meta.json (first-seen batch id). Loading de-duplicates by event_id, so a replayed
    batch that appends the same events twice is harmless.
    """

    def __init__(self, state_dir: str, ttl_batches: int):
        self.dir = state_dir
        self.ttl = ttl_batches
        os.makedirs(state_dir, exist_ok=True)

    def _p(self, snap: str, suffix: str) -> str:
        return os.path.join(self.dir, snap.replace(":", "_") + suffix)

    def add(self, snap: str, raw_events: list, batch_id: int):
        with open(self._p(snap, ".jsonl"), "a", encoding="utf-8") as f:
            for r in raw_events:
                f.write(r + "\n")
        meta = self._p(snap, ".meta.json")
        if not os.path.exists(meta):
            json.dump({"snapshot_id": snap, "first_batch": batch_id}, open(meta, "w"))

    def load(self, snap: str) -> Dict[str, str]:
        out = {}
        for line in open(self._p(snap, ".jsonl"), encoding="utf-8"):
            out[json.loads(line)["event_id"]] = line.strip()
        return out

    def pending(self):
        for meta in glob.glob(os.path.join(self.dir, "*.meta.json")):
            yield json.load(open(meta))

    def drop(self, snap: str):
        for suffix in (".jsonl", ".meta.json"):
            if os.path.exists(self._p(snap, suffix)):
                os.remove(self._p(snap, suffix))


class BatchProcessor:
    def __init__(self, service: ScoringService, out_dir: str, snapshot_ttl_batches: int = 20):
        self.service = service
        self.out = out_dir
        self.buffer = SnapshotBuffer(os.path.join(out_dir, "state", "elliptic_snapshots"), snapshot_ttl_batches)
        self.graph_mode = bool(service.elliptic and service.elliptic.requires_graph)
        self.seen_index = os.path.join(out_dir, "state", "scored_event_ids.txt")
        os.makedirs(os.path.dirname(self.seen_index), exist_ok=True)
        self.seen = set(open(self.seen_index).read().split()) if os.path.exists(self.seen_index) else set()
        for d in ("predictions", "dead_letter", "monitoring"):
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    # ------------------------------------------------------------------ main
    def process(self, pdf: pd.DataFrame, batch_id: int) -> Dict:
        t_start = time.perf_counter()
        processing_time = utc_now_iso()
        dlq_rows, raw_ok, meta = [], [], {}
        for r in pdf.itertuples(index=False):
            kts = r.kafka_timestamp.isoformat() if pd.notna(r.kafka_timestamp) else None
            if r.event_id is None or pd.isna(r.event_id):
                dlq_rows.append((r.kafka_partition, r.kafka_offset, r.key, r.value, "unparseable_json_or_missing_event_id"))
                continue
            raw_ok.append(r.value)
            meta[r.event_id] = (r.kafka_partition, r.kafka_offset, kts)

        to_score, buffered, dup_sink = [], 0, 0
        if self.graph_mode:
            by_snap = {}
            for v in raw_ok:
                try:
                    obj = json.loads(v)
                except ValueError:
                    obj = {}
                snap = (obj.get("graph") or {}).get("snapshot_id") if obj.get("source_dataset") == "elliptic" else None
                if snap and obj.get("event_id") in self.seen:
                    # already scored in a released snapshot: buffering it again would open a snapshot
                    # that can never complete and later expire as a spurious rejection
                    dup_sink += 1
                elif snap:
                    by_snap.setdefault(snap, []).append(v)
                else:
                    to_score.append(v)
            for snap, vals in by_snap.items():
                self.buffer.add(snap, vals, batch_id)
                buffered += len(vals)
        else:
            to_score = raw_ok

        results = self.service.score_events(to_score, processing_time=processing_time)
        if self.graph_mode:
            results.extend(self._release_snapshots(batch_id, processing_time))

        pred_rows = []
        for res in results:
            if res.status == "rejected_invalid_event":
                p, o, _ = meta.get(res.event_id, (None, None, None))
                dlq_rows.append((p, o, res.event_id, None, res.error))
                continue
            if res.status == "scored" and res.event_id in self.seen:
                dup_sink += 1  # duplicate that escaped the streaming watermark
                continue
            pred_rows.append(self._row(res, meta.get(res.event_id, (None, None, None)), batch_id))

        if pred_rows:
            _atomic_parquet(pa.Table.from_pylist(pred_rows, schema=PRED_SCHEMA),
                            os.path.join(self.out, "predictions", f"batch={batch_id:08d}.parquet"))
            new_ids = [r["event_id"] for r in pred_rows if r["status"] == "scored"]
            self.seen.update(new_ids)
            with open(self.seen_index, "a") as f:
                f.write("\n".join(new_ids) + ("\n" if new_ids else ""))
        if dlq_rows:
            now = utc_now_iso()

            def s(x):  # pandas hands missing strings over as NaN floats
                return None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)

            def n(x):
                return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)

            _atomic_parquet(pa.Table.from_pylist(
                [{"kafka_partition": n(p), "kafka_offset": n(o), "key": s(k), "value": s(v), "reason": s(why),
                  "batch_id": batch_id, "detected_at": now} for p, o, k, v, why in dlq_rows], schema=DLQ_SCHEMA),
                os.path.join(self.out, "dead_letter", f"batch={batch_id:08d}.parquet"))
        lat = [r["e2e_latency_ms"] for r in pred_rows if r["e2e_latency_ms"] is not None]
        status_counts = pd.Series([r["status"] for r in pred_rows]).value_counts().to_dict() if pred_rows else {}
        m = {"batch_id": batch_id, "processing_time": processing_time, "input_rows": len(pdf),
             "dead_lettered": len(dlq_rows), "buffered_for_graph_context": buffered,
             "results_written": len(pred_rows), "status_counts": status_counts,
             "duplicates_dropped_at_sink": dup_sink, "flagged": int(sum(bool(r["flagged"]) for r in pred_rows)),
             "batch_seconds": round(time.perf_counter() - t_start, 4),
             "e2e_latency_ms_p50": float(np.percentile(lat, 50)) if lat else None,
             "e2e_latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
             "e2e_latency_ms_max": float(np.max(lat)) if lat else None}
        with open(os.path.join(self.out, "monitoring", "batches.jsonl"), "a") as f:
            f.write(json.dumps(m) + "\n")
        log.info("batch %s: %s", batch_id, m)
        return m

    def _release_snapshots(self, batch_id: int, processing_time: str):
        out = []
        for meta in list(self.buffer.pending()):
            snap = meta["snapshot_id"]
            events = self.buffer.load(snap)
            first = json.loads(next(iter(events.values())))
            expected = first["graph"]["snapshot_size"]
            if len(events) >= expected:
                out.extend(self.service.score_events(list(events.values()), processing_time=processing_time))
                self.buffer.drop(snap)
            elif batch_id - meta["first_batch"] >= self.buffer.ttl:
                # state expiry: report explicitly instead of guessing a graph context
                for v in events.values():
                    ev = EventEnvelope.model_validate({"event": json.loads(v)}).event
                    out.append(PredictionResult(event_id=ev.event_id, transaction_id=ev.transaction_id,
                                                source_dataset="elliptic", event_time=ev.event_time,
                                                status="rejected_incomplete_graph_context",
                                                error=f"{snap} expired after {self.buffer.ttl} batches with "
                                                      f"{len(events)}/{expected} transactions",
                                                produced_at=ev.produced_at, processing_time=processing_time))
                self.buffer.drop(snap)
        return out

    @staticmethod
    def _row(res: PredictionResult, meta, batch_id) -> Dict:
        part, off, kts = meta
        prod = _iso_to_epoch_ms(res.produced_at)
        scored = _iso_to_epoch_ms(res.scored_at)
        return {"event_id": res.event_id, "transaction_id": res.transaction_id, "source_dataset": res.source_dataset,
                "event_time_kind": res.event_time.kind if res.event_time else None,
                "event_time_value": res.event_time.value if res.event_time else None,
                "model_id": res.model_id, "model_version": res.model_version, "score": res.score,
                "score_type": res.score_type, "score_percentile": res.score_percentile, "threshold": res.threshold,
                "flagged": res.flagged, "reason_codes": ";".join(res.reason_codes), "graph_context": res.graph_context,
                "status": res.status, "error": res.error, "produced_at": res.produced_at, "kafka_timestamp": kts,
                "processing_time": res.processing_time, "scored_at": res.scored_at, "kafka_partition": part,
                "kafka_offset": off, "batch_id": batch_id,
                "e2e_latency_ms": (scored - prod) if (prod is not None and scored is not None) else None}


def build_query(spark, bootstrap: str, topic: str, out_dir: str, checkpoint: str, processor: BatchProcessor,
                trigger: str = "availableNow", max_offsets_per_trigger: int = 5000,
                dedup_watermark: str = "30 minutes", starting_offsets: str = "earliest"):
    import pyspark.sql.functions as F
    from pyspark.sql.types import StringType, StructField, StructType

    from src.spark.session import to_spark_path

    env_schema = StructType([StructField("schema_version", StringType()), StructField("event_id", StringType()),
                             StructField("source_dataset", StringType())])
    raw = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", bootstrap)
           .option("subscribe", topic).option("startingOffsets", starting_offsets)
           .option("maxOffsetsPerTrigger", max_offsets_per_trigger).option("failOnDataLoss", "true").load())
    parsed = (raw.select(F.col("key").cast("string").alias("key"), F.col("value").cast("string").alias("value"),
                         F.col("partition").alias("kafka_partition"), F.col("offset").alias("kafka_offset"),
                         F.col("timestamp").alias("kafka_timestamp"))
              .withColumn("env", F.from_json("value", env_schema))
              .withColumn("event_id", F.col("env.event_id"))
              .withColumn("dedup_key", F.coalesce(F.col("event_id"),
                                                  F.concat_ws(":", F.lit("malformed"), F.col("kafka_partition"),
                                                              F.col("kafka_offset"))))
              .drop("env")
              .withWatermark("kafka_timestamp", dedup_watermark)
              .dropDuplicatesWithinWatermark(["dedup_key"]))

    def foreach_batch(df, batch_id):
        pdf = df.select("key", "value", "kafka_partition", "kafka_offset", "kafka_timestamp", "event_id").toPandas()
        if len(pdf):
            pdf = pdf.sort_values(["kafka_partition", "kafka_offset"]).reset_index(drop=True)
        processor.process(pdf, int(batch_id))

    writer = (parsed.writeStream.foreachBatch(foreach_batch)
              .option("checkpointLocation", to_spark_path(checkpoint)).queryName("bsad_scoring"))
    if trigger == "availableNow":
        writer = writer.trigger(availableNow=True)
    else:
        writer = writer.trigger(processingTime=trigger)
    return writer.start()


def progress_listener(path: str):
    from pyspark.sql.streaming import StreamingQueryListener

    class _L(StreamingQueryListener):
        def onQueryStarted(self, event):
            pass

        def onQueryProgress(self, event):
            p = json.loads(event.progress.json)
            dropped = sum((s.get("customMetrics") or {}).get("numDroppedDuplicateRows", 0)
                          for s in p.get("stateOperators", []))
            rec = {"batch_id": p.get("batchId"), "timestamp": p.get("timestamp"),
                   "num_input_rows": p.get("numInputRows"), "input_rows_per_second": p.get("inputRowsPerSecond"),
                   "processed_rows_per_second": p.get("processedRowsPerSecond"),
                   "duration_ms": p.get("durationMs"), "dropped_duplicates_streaming": dropped,
                   "state_rows": sum(s.get("numRowsTotal", 0) for s in p.get("stateOperators", [])),
                   "watermark": (p.get("eventTime") or {}).get("watermark"),
                   "sources": [{"start": s.get("startOffset"), "end": s.get("endOffset")} for s in p.get("sources", [])]}
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        def onQueryTerminated(self, event):
            with open(path, "a") as f:
                f.write(json.dumps({"terminated": str(event.id), "exception": event.exception}) + "\n")

        def onQueryIdle(self, event):
            pass

    return _L()
