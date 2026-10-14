"""
Adapters: dataset rows -> canonical events -> model inputs.
"""
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from src.contracts.schemas import (DataQuality, EllipticEvent, ZenodoEvent, utc_now_iso)

ELLIPTIC_FEATURES = [f"f{i}" for i in range(1, 166)]
ZENODO_ATTRS = ["block_number", "tx_index", "num_inputs", "num_outputs", "total_input_satoshi",
                "total_output_satoshi", "fee_satoshi", "fee_rate", "is_coinbase", "avg_input_val", "avg_output_val",
                "input_output_count_ratio", "input_output_val_ratio", "max_input_prior_tx_count",
                "avg_input_prior_tx_count", "max_input_prior_volume_satoshi", "new_address_fraction"]


# ------------------------------------------------------------------ Zenodo
def zenodo_row_to_event(row: Dict, seq: Optional[int] = None, produced_at: Optional[str] = None) -> Dict:
    """Map one row of data/parquet/zenodo_features to a ZenodoEvent dict (validated)."""
    tx = str(row["tx_hash"])
    attrs = {k: (int(row[k]) if k in ("block_number", "tx_index", "num_inputs", "num_outputs", "total_input_satoshi",
                                      "total_output_satoshi", "fee_satoshi", "is_coinbase",
                                      "max_input_prior_tx_count", "max_input_prior_volume_satoshi")
                 else float(row[k])) for k in ZENODO_ATTRS}
    ev = ZenodoEvent(source_dataset="zenodo", event_id=f"zenodo:{tx}", transaction_id=tx, tx_hash=tx,
                     event_time={"kind": "block_height", "value": attrs["block_number"]},
                     produced_at=produced_at or utc_now_iso(), replay_sequence=seq, attributes=attrs)
    return ev.model_dump(mode="json")


def zenodo_events_to_frame(events: List[ZenodoEvent]) -> pd.DataFrame:
    rows = []
    for e in events:
        r = e.attributes.model_dump()
        r["tx_hash"] = e.tx_hash
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Elliptic
def elliptic_snapshot_events(nodes: pd.DataFrame, edges: pd.DataFrame, start_seq: int = 0,
                             produced_at: Optional[str] = None) -> Iterator[Dict]:
    """Yield EllipticEvent dicts for one or more complete time-step snapshots.

    ``nodes``: tx_id, time_step, f1..f165 (raw). ``edges``: source_tx_id, target_tx_id.
    Labels are deliberately NOT part of the event (the scorer must never see them).
    """
    sizes = nodes.groupby("time_step").size().to_dict()
    out_nb = edges.groupby("source_tx_id")["target_tx_id"].apply(list).to_dict()
    seq = start_seq
    for row in nodes.sort_values(["time_step", "tx_id"]).itertuples(index=False):
        rd = row._asdict()
        tx = str(int(rd["tx_id"]))
        t = int(rd["time_step"])
        ev = EllipticEvent(
            source_dataset="elliptic", event_id=f"elliptic:{tx}", transaction_id=tx,
            event_time={"kind": "elliptic_time_step", "value": t}, produced_at=produced_at or utc_now_iso(),
            replay_sequence=seq, attributes={"features": [float(rd[c]) for c in ELLIPTIC_FEATURES]},
            graph={"snapshot_id": f"elliptic:step:{t}", "snapshot_size": int(sizes[t]),
                   "out_neighbors": [str(int(n)) for n in out_nb.get(rd["tx_id"], [])]},
            data_quality=DataQuality())
        seq += 1
        yield ev.model_dump(mode="json")


def elliptic_events_to_snapshot(events: List[EllipticEvent]):
    """Rebuild (nodes, edges) model input from events of ONE snapshot."""
    nodes = pd.DataFrame(np.array([e.attributes.features for e in events], dtype=np.float64),
                         columns=ELLIPTIC_FEATURES)
    nodes.insert(0, "tx_id", [int(e.transaction_id) for e in events])
    src, dst = [], []
    for e in events:
        for n in e.graph.out_neighbors:
            src.append(int(e.transaction_id))
            dst.append(int(n))
    edges = pd.DataFrame({"source_tx_id": np.array(src, dtype=np.int64), "target_tx_id": np.array(dst, dtype=np.int64)})
    return nodes, edges
