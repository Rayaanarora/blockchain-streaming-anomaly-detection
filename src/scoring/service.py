"""
Unified scoring interface with explicit, dataset-based routing.

  source_dataset == "elliptic" -> supervised Elliptic bundle (illicit_probability)
  source_dataset == "zenodo"   -> Zenodo Isolation Forest detector (isolation_forest_anomaly_score)

The two scores have different meanings and calibrations; every result carries
``score_type``, ``model_id`` and ``model_version`` so they are never mixed up. No event is
ever fed to the other dataset's model: routing is decided by the validated event type.
"""
import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
from pydantic import ValidationError

from src.contracts.adapters import elliptic_events_to_snapshot, zenodo_events_to_frame
from src.contracts.schemas import EllipticEvent, EventEnvelope, PredictionResult, ZenodoEvent, utc_now_iso

log = logging.getLogger("bsad.scoring")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class ScoringConfig:
    """Paths come from environment variables (see .env.example) with repository defaults."""

    def __init__(self, elliptic_bundle_dir: Optional[str] = None, zenodo_model_dir: Optional[str] = None):
        from src.models.elliptic_bundle import latest_bundle_dir
        e_root = os.environ.get("BSAD_ELLIPTIC_MODEL_ROOT", os.path.join(ROOT, "artifacts/models/elliptic"))
        z_root = os.environ.get("BSAD_ZENODO_MODEL_ROOT", os.path.join(ROOT, "artifacts/models/zenodo"))
        self.elliptic_bundle_dir = elliptic_bundle_dir or os.environ.get("BSAD_ELLIPTIC_BUNDLE") or \
            (latest_bundle_dir(e_root) if os.path.isdir(e_root) else None)
        self.zenodo_model_dir = zenodo_model_dir or os.environ.get("BSAD_ZENODO_MODEL") or \
            (latest_bundle_dir(z_root) if os.path.isdir(z_root) else None)


class ScoringService:
    def __init__(self, config: Optional[ScoringConfig] = None, load_elliptic: bool = True, load_zenodo: bool = True):
        self.config = config or ScoringConfig()
        self.elliptic = self.zenodo = None
        self.load_seconds = {}
        if load_elliptic and self.config.elliptic_bundle_dir:
            from src.models.elliptic_bundle import EllipticModelBundle
            t0 = time.perf_counter()
            self.elliptic = EllipticModelBundle(self.config.elliptic_bundle_dir)
            self.load_seconds["elliptic"] = time.perf_counter() - t0
        if load_zenodo and self.config.zenodo_model_dir:
            from src.anomaly.zenodo_detector import ZenodoAnomalyDetector
            t0 = time.perf_counter()
            self.zenodo = ZenodoAnomalyDetector.load(self.config.zenodo_model_dir)
            self.load_seconds["zenodo"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ info
    def models(self) -> Dict:
        out = {}
        if self.elliptic:
            m = self.elliptic.manifest
            out["elliptic"] = {"model_id": m["model_id"], "model_version": m["model_version"], "config": m["config"],
                               "score_type": m["score_type"], "threshold": m["decision_threshold"],
                               "requires_graph": m["requires_graph"], "score_semantics": m["score_semantics"]}
        if self.zenodo:
            m = self.zenodo.manifest
            out["zenodo"] = {"model_id": m["model_id"], "model_version": m["model_version"],
                             "score_type": m["score_type"], "threshold": m["threshold"],
                             "feature_version": m["feature_version"], "score_semantics": m["score_semantics"]}
        return out

    # ------------------------------------------------------------------ entry point
    def score_events(self, raw_events: Iterable, processing_time: Optional[str] = None) -> List[PredictionResult]:
        """Validate, route and score a batch of events (dicts, JSON strings or parsed models)."""
        processing_time = processing_time or utc_now_iso()
        results: List[PredictionResult] = []
        elliptic: List[EllipticEvent] = []
        zenodo: List[ZenodoEvent] = []
        for raw in raw_events:
            try:
                if isinstance(raw, (EllipticEvent, ZenodoEvent)):
                    ev = raw
                else:
                    payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
                    ev = EventEnvelope.model_validate({"event": payload}).event
            except (ValidationError, ValueError, TypeError) as exc:
                eid = raw.get("event_id") if isinstance(raw, dict) else None
                results.append(PredictionResult(event_id=eid, status="rejected_invalid_event",
                                                error=_short_error(exc), processing_time=processing_time))
                continue
            (elliptic if ev.source_dataset == "elliptic" else zenodo).append(ev)
        if zenodo:
            results.extend(self._score_zenodo(zenodo, processing_time))
        if elliptic:
            results.extend(self._score_elliptic(elliptic, processing_time))
        return results

    # ------------------------------------------------------------------ zenodo
    def _base(self, ev, processing_time):
        return dict(event_id=ev.event_id, transaction_id=ev.transaction_id, source_dataset=ev.source_dataset,
                    event_time=ev.event_time, produced_at=ev.produced_at, processing_time=processing_time)

    def _score_zenodo(self, events: List[ZenodoEvent], processing_time) -> List[PredictionResult]:
        from src.anomaly.zenodo_features import compute_features, dq_status
        if self.zenodo is None:
            return [PredictionResult(**self._base(e, processing_time), status="error",
                                     error="zenodo detector not loaded") for e in events]
        m = self.zenodo.manifest
        common = dict(model_id=m["model_id"], model_version=m["model_version"],
                      score_type="isolation_forest_anomaly_score", threshold=self.zenodo.threshold,
                      graph_context="not_available")
        df = zenodo_events_to_frame(events)
        dq = dq_status(df)
        X = compute_features(df).to_numpy(np.float64)
        ok = (dq == "ok") & (df["is_coinbase"].to_numpy() == 0) & np.isfinite(X).all(axis=1)
        scored = self.zenodo.score(X[ok]) if ok.any() else None
        out, k = [], 0
        for i, ev in enumerate(events):
            base = {**self._base(ev, processing_time), **common}
            if dq[i] != "ok":
                out.append(PredictionResult(**base, status="skipped_invalid_input", error=dq[i]))
            elif df["is_coinbase"].iat[i] == 1:
                out.append(PredictionResult(**base, status="skipped_coinbase"))
            elif not ok[i]:
                out.append(PredictionResult(**base, status="skipped_invalid_input", error="non-finite features"))
            else:
                out.append(PredictionResult(**base, status="scored", score=float(scored["score"][k]),
                                            score_percentile=float(scored["percentile"][k]),
                                            flagged=bool(scored["flagged"][k]),
                                            reason_codes=scored["reason_codes"][k].split(";")))
                k += 1
        return out

    # ------------------------------------------------------------------ elliptic
    def _score_elliptic(self, events: List[EllipticEvent], processing_time) -> List[PredictionResult]:
        if self.elliptic is None:
            return [PredictionResult(**self._base(e, processing_time), status="error",
                                     error="elliptic model not loaded") for e in events]
        b = self.elliptic
        common = dict(model_id=b.model_id, model_version=b.version, score_type="illicit_probability",
                      threshold=b.threshold)
        groups = defaultdict(dict)
        for e in events:  # de-duplicate within a snapshot by transaction id (last wins)
            groups[e.graph.snapshot_id][e.transaction_id] = e
        out = []
        for snap_id, by_tx in groups.items():
            evs = list(by_tx.values())
            expected = evs[0].graph.snapshot_size
            if b.requires_graph and len(evs) != expected:
                out.extend(PredictionResult(**self._base(e, processing_time), **common,
                                            status="rejected_incomplete_graph_context",
                                            error=f"{snap_id}: have {len(evs)} of {expected} transactions; "
                                                  "graph model needs the complete snapshot")
                           for e in evs)
                continue
            nodes, edges = elliptic_events_to_snapshot(evs)
            try:
                pred = b.predict(nodes, edges if b.requires_graph else None)
            except Exception as exc:  # validation errors are reported per event, not swallowed silently
                log.exception("elliptic scoring failed for %s", snap_id)
                out.extend(PredictionResult(**self._base(e, processing_time), **common, status="error",
                                            error=_short_error(exc)) for e in evs)
                continue
            for e, (_, r) in zip(evs, pred.iterrows()):
                out.append(PredictionResult(**self._base(e, processing_time), **common, status="scored",
                                            score=float(r["score"]), flagged=bool(r["flagged"]),
                                            graph_context=r["graph_context"]))
        return out


def _short_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    return f"{type(exc).__name__}: {text[:300]}"
