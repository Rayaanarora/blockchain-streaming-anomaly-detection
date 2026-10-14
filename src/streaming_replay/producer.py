"""
Deterministic replay producer: existing Parquet data -> canonical events -> Kafka.

* Order: Zenodo rows by (block_number, tx_index); Elliptic nodes by (time_step, tx_id).
* Stable event ids ("<dataset>:<transaction id>") are also the Kafka message key, so all
  versions of an event land in the same partition.
* Every event is validated against the contract before sending (invalid source rows are
  counted and NOT sent, unless explicitly injected for tests).
* Idempotent Kafka producer (enable.idempotence) avoids broker-side duplicates from retries;
  delivery failures are counted and make the run fail.
* ``inject_malformed`` / ``inject_duplicates`` exist ONLY to exercise the consumer's
  dead-letter and de-duplication paths in tests; they are reported in the run manifest.
"""
import json
import os
import time
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd
import pyarrow.dataset as ds

from src.contracts.adapters import elliptic_snapshot_events, zenodo_row_to_event
from src.contracts.schemas import EventEnvelope, utc_now_iso

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ZENODO_COLUMNS = ["block_number", "tx_index", "tx_hash", "num_inputs", "num_outputs", "total_input_satoshi",
                  "total_output_satoshi", "fee_satoshi", "fee_rate", "is_coinbase", "avg_input_val", "avg_output_val",
                  "input_output_count_ratio", "input_output_val_ratio", "max_input_prior_tx_count",
                  "avg_input_prior_tx_count", "max_input_prior_volume_satoshi", "new_address_fraction"]


def zenodo_source(parquet_dir: str, block_chunk: str, limit: int, start_block: Optional[int] = None) -> pd.DataFrame:
    """Bounded read of one existing Zenodo partition (first ``limit`` rows in chain order)."""
    dset = ds.dataset(os.path.join(parquet_dir, f"block_chunk={block_chunk}"), format="parquet")
    flt = ds.field("block_number") >= start_block if start_block is not None else None
    rows, frames = 0, []
    for batch in dset.to_batches(columns=ZENODO_COLUMNS, filter=flt, batch_size=min(limit, 50_000)):
        frames.append(batch.to_pandas())
        rows += batch.num_rows
        if rows >= limit:
            break
    df = pd.concat(frames).sort_values(["block_number", "tx_index"]).head(limit).reset_index(drop=True)
    return df


def zenodo_events(df: pd.DataFrame, start_seq: int = 0) -> Iterator[Dict]:
    for i, row in enumerate(df.to_dict("records")):
        yield zenodo_row_to_event(row, seq=start_seq + i)


def elliptic_events(spark_dir: str, steps: List[int], start_seq: int = 0) -> Iterator[Dict]:
    """Complete snapshots of the given time steps from the Spark-produced Elliptic dataset."""
    nodes = pd.read_parquet(os.path.join(spark_dir, "nodes"), filters=[("time_step", "in", steps)])
    nodes["time_step"] = nodes["time_step"].astype(int)
    edges = pd.read_parquet(os.path.join(spark_dir, "edges"), filters=[("time_step", "in", steps)])
    yield from elliptic_snapshot_events(nodes.drop(columns=["label", "label_status", "split", "node_idx"]), edges,
                                        start_seq=start_seq)


class ReplayProducer:
    def __init__(self, bootstrap: str, topic: str, client_id: str = "bsad-replay-producer"):
        from confluent_kafka import Producer
        self.topic = topic
        self.producer = Producer({"bootstrap.servers": bootstrap, "client.id": client_id,
                                  "enable.idempotence": True, "acks": "all", "linger.ms": 20,
                                  "compression.type": "lz4", "message.timeout.ms": 60000})
        self.delivered = 0
        self.failed: List[str] = []

    def _cb(self, err, msg):
        if err is not None:
            self.failed.append(f"{msg.key()}: {err}")
        else:
            self.delivered += 1

    def _send(self, key: str, value: bytes):
        while True:
            try:
                self.producer.produce(self.topic, key=key.encode(), value=value, on_delivery=self._cb)
                break
            except BufferError:
                self.producer.poll(0.05)
        self.producer.poll(0)

    def replay(self, events: Iterable[Dict], rate: float = 0.0, inject_malformed: int = 0,
               inject_duplicates: int = 0, batch_flush: int = 5000) -> Dict:
        """Send events. ``rate`` = events/second (0 = as fast as possible)."""
        t0 = time.perf_counter()
        sent, invalid, first, last = 0, 0, None, None
        dup_pool: List[tuple] = []
        for ev in events:
            try:
                EventEnvelope.model_validate({"event": ev})
            except Exception:
                invalid += 1
                continue
            ev["produced_at"] = utc_now_iso()
            value = json.dumps(ev, separators=(",", ":")).encode()
            self._send(ev["event_id"], value)
            if len(dup_pool) < inject_duplicates:
                dup_pool.append((ev["event_id"], value))
            sent += 1
            first = first if first is not None else ev.get("replay_sequence")
            last = ev.get("replay_sequence")
            if sent % batch_flush == 0:
                self.producer.flush(30)
            if rate > 0:
                target = t0 + sent / rate
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        for key, value in dup_pool:  # exact re-sends of already-sent events (test only)
            self._send(key, value)
        for i in range(inject_malformed):  # test only: not JSON / schema-violating
            self._send(f"malformed:{i}", b"{this is not json" if i % 2 == 0 else
                       json.dumps({"schema_version": "1.0", "source_dataset": "zenodo", "event_id": f"bad:{i}"}).encode())
        remaining = self.producer.flush(60)
        elapsed = time.perf_counter() - t0
        if remaining or self.failed:
            raise RuntimeError(f"{remaining} undelivered, {len(self.failed)} failed: {self.failed[:3]}")
        return {"topic": self.topic, "events_sent": sent, "invalid_source_rows_not_sent": invalid,
                "duplicates_injected": len(dup_pool), "malformed_injected": inject_malformed,
                "messages_delivered": self.delivered, "first_sequence": first, "last_sequence": last,
                "elapsed_seconds": round(elapsed, 3), "send_rate_eps": round(self.delivered / max(elapsed, 1e-9), 1),
                "requested_rate_eps": rate}
