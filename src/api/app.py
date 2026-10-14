"""
Backend API + dashboard for persisted pipeline outputs.

Everything shown comes from files written by the pipeline (no fabricated data):
  streaming run dir  (default data/streaming/demo, env BSAD_STREAM_RUN_DIR or ?run=<name>)
     predictions/*.parquet, dead_letter/*.parquet, monitoring/{batches,progress}.jsonl, monitoring/run_info.json
  batch outputs      data/processed/zenodo_anomaly/{score_summary.json,top_ranked.parquet}
                     data/validation/zenodo_anomaly/evaluation.json
                     data/validation/final_model_selection/{summary.csv,selection.json}
  model manifests    artifacts/models/{elliptic,zenodo}/<version>/manifest.json

Start:  .venv\\Scripts\\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
Open:   http://127.0.0.1:8000/
"""
import glob
import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STREAM_ROOT = os.path.join(ROOT, "data", "streaming")
STATIC = os.path.join(os.path.dirname(__file__), "static")
log = logging.getLogger("bsad.api")


def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    log.warning("skipping corrupt monitoring line in %s", path)
    return out


def _read_parquet_dir(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _json(path: str):
    return json.load(open(path)) if os.path.exists(path) else None


def _clean(obj):
    """Make pandas/numpy values JSON-safe (NaN -> None)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def create_app(run_dir: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Blockchain Streaming Anomaly Detection", version="1.0.0")
    default_run = run_dir or os.environ.get("BSAD_STREAM_RUN_DIR") or os.path.join(STREAM_ROOT, "demo")
    state = {"service": None}

    def resolve(run: Optional[str]) -> str:
        if not run:
            return default_run
        if os.path.sep in run or "/" in run or ".." in run:
            raise HTTPException(400, "run must be a directory name under data/streaming")
        path = os.path.join(STREAM_ROOT, run)
        if not os.path.isdir(path):
            raise HTTPException(404, f"unknown run '{run}'")
        return path

    def predictions(run_path: str) -> pd.DataFrame:
        return _read_parquet_dir(os.path.join(run_path, "predictions"))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/runs")
    def runs():
        items = []
        for d in sorted(glob.glob(os.path.join(STREAM_ROOT, "*"))):
            if os.path.isdir(os.path.join(d, "predictions")):
                files = glob.glob(os.path.join(d, "predictions", "*.parquet"))
                items.append({"name": os.path.basename(d), "prediction_files": len(files),
                              "last_modified": max((os.path.getmtime(f) for f in files), default=None)})
        return {"default": os.path.basename(default_run), "runs": items}

    @app.get("/api/health")
    def health(run: Optional[str] = None):
        rp = resolve(run)
        batches = _read_jsonl(os.path.join(rp, "monitoring", "batches.jsonl"))
        progress = [p for p in _read_jsonl(os.path.join(rp, "monitoring", "progress.jsonl")) if "batch_id" in p]
        terminated = [p for p in _read_jsonl(os.path.join(rp, "monitoring", "progress.jsonl")) if "terminated" in p]
        kafka = None
        try:
            from confluent_kafka.admin import AdminClient
            AdminClient({"bootstrap.servers": os.environ.get("BSAD_KAFKA_BOOTSTRAP", "127.0.0.1:29092")}).list_topics(timeout=1.5)
            kafka = "reachable"
        except Exception:
            kafka = "unreachable"
        last_exc = terminated[-1]["exception"] if terminated else None
        status = "no_data" if not batches else ("error" if last_exc else "ok")
        return _clean({"status": status, "run_dir": rp, "kafka": kafka, "batches_recorded": len(batches),
                       "last_batch_processing_time": batches[-1]["processing_time"] if batches else None,
                       "last_query_progress_epoch_ms": progress[-1]["timestamp"] if progress else None,
                       "query_terminated": bool(terminated), "last_query_exception": last_exc,
                       "run_info": _json(os.path.join(rp, "monitoring", "run_info.json"))})

    @app.get("/api/summary")
    def summary(run: Optional[str] = None):
        rp = resolve(run)
        p = predictions(rp)
        dlq = _read_parquet_dir(os.path.join(rp, "dead_letter"))
        batches = _read_jsonl(os.path.join(rp, "monitoring", "batches.jsonl"))
        progress = [x for x in _read_jsonl(os.path.join(rp, "monitoring", "progress.jsonl")) if "batch_id" in x]
        out = {"run": os.path.basename(rp), "predictions_total": int(len(p)), "dead_letter_total": int(len(dlq)),
               "duplicates_dropped_streaming": int(sum(x.get("dropped_duplicates_streaming") or 0 for x in progress)),
               "duplicates_dropped_at_sink": int(sum(b.get("duplicates_dropped_at_sink", 0) for b in batches)),
               "batches": len(batches)}
        if len(p):
            out["by_dataset"] = {}
            for ds_name, g in p.groupby("source_dataset"):
                out["by_dataset"][ds_name] = {
                    "rows": int(len(g)), "scored": int((g.status == "scored").sum()),
                    "flagged": int(g.flagged.fillna(False).sum()),
                    "status_counts": g.status.value_counts().to_dict(),
                    "model_id": g.model_id.dropna().iloc[0] if g.model_id.notna().any() else None,
                    "model_version": g.model_version.dropna().iloc[0] if g.model_version.notna().any() else None,
                    "score_type": g.score_type.dropna().iloc[0] if g.score_type.notna().any() else None,
                    "threshold": g.threshold.dropna().iloc[0] if g.threshold.notna().any() else None}
            lat = p.e2e_latency_ms.dropna()
            out["errors_total"] = int((p.status == "error").sum())
            out["e2e_latency_ms"] = {"p50": lat.quantile(0.5), "p95": lat.quantile(0.95), "max": lat.max(),
                                     "definition": "scored_at - produced_at (producer send -> prediction written; "
                                                   "includes Kafka, Spark micro-batch wait and scoring)"} if len(lat) else None
            ts = pd.to_datetime(p.scored_at, utc=True)
            span = (ts.max() - ts.min()).total_seconds()
            out["throughput"] = {"rows_per_second_over_run": (len(p) / span) if span > 0 else None,
                                 "first_scored_at": ts.min().isoformat(), "last_scored_at": ts.max().isoformat()}
        if progress:
            prs = [x["processed_rows_per_second"] for x in progress if x.get("num_input_rows")]
            out["spark_processed_rows_per_second_median"] = float(np.median(prs)) if prs else None
        if batches:
            out["batch_seconds_median"] = float(np.median([b["batch_seconds"] for b in batches]))
        return _clean(out)

    @app.get("/api/predictions")
    def recent(run: Optional[str] = None, limit: int = Query(50, ge=1, le=1000), dataset: Optional[str] = None,
               flagged_only: bool = False, status: Optional[str] = None):
        p = predictions(resolve(run))
        if len(p) == 0:
            return {"items": [], "total": 0}
        if dataset:
            p = p[p.source_dataset == dataset]
        if flagged_only:
            p = p[p.flagged.fillna(False)]
        if status:
            p = p[p.status == status]
        total = len(p)
        p = p.sort_values(["scored_at", "kafka_offset"], ascending=False).head(limit)
        return _clean({"items": p.replace({np.nan: None}).to_dict("records"), "total": int(total)})

    @app.get("/api/distribution")
    def distribution(run: Optional[str] = None, bins: int = Query(30, ge=5, le=100)):
        p = predictions(resolve(run))
        out = {}
        if len(p):
            for (ds_name, st), g in p[p.status == "scored"].groupby(["source_dataset", "score_type"]):
                s = g.score.to_numpy(float)
                lo, hi = (0.0, 1.0) if st == "illicit_probability" else (float(s.min()), float(s.max()) + 1e-9)
                counts, edges = np.histogram(s, bins=bins, range=(lo, hi))
                out[ds_name] = {"score_type": st, "counts": counts.tolist(), "edges": edges.tolist(),
                                "threshold": float(g.threshold.iloc[0]), "n": int(len(s))}
        return _clean(out)

    @app.get("/api/batches")
    def batches_(run: Optional[str] = None, limit: int = 200):
        rp = resolve(run)
        return _clean({"batches": _read_jsonl(os.path.join(rp, "monitoring", "batches.jsonl"))[-limit:],
                       "progress": [x for x in _read_jsonl(os.path.join(rp, "monitoring", "progress.jsonl"))
                                    if "batch_id" in x][-limit:]})

    @app.get("/api/dead-letter")
    def dead_letter(run: Optional[str] = None, limit: int = 50):
        d = _read_parquet_dir(os.path.join(resolve(run), "dead_letter"))
        if len(d) == 0:
            return {"items": [], "total": 0}
        d["value"] = d["value"].fillna("").str.slice(0, 200)
        return _clean({"items": d.tail(limit).replace({np.nan: None}).to_dict("records"), "total": int(len(d))})

    @app.get("/api/batch-results")
    def batch_results():
        """Offline (batch) results: Zenodo full-dataset scoring and Elliptic model selection."""
        z = _json(os.path.join(ROOT, "data/processed/zenodo_anomaly/score_summary.json"))
        top_path = os.path.join(ROOT, "data/processed/zenodo_anomaly/top_ranked.parquet")
        top = pd.read_parquet(top_path).head(25).replace({np.nan: None}).to_dict("records") if os.path.exists(top_path) else []
        sel_csv = os.path.join(ROOT, "data/validation/final_model_selection/summary.csv")
        sel = pd.read_csv(sel_csv).replace({np.nan: None}).to_dict("records") if os.path.exists(sel_csv) else []
        return _clean({"zenodo_score_summary": z, "zenodo_top_ranked": top,
                       "zenodo_evaluation": _json(os.path.join(ROOT, "data/validation/zenodo_anomaly/evaluation.json")),
                       "elliptic_model_selection": sel,
                       "elliptic_selection": _json(os.path.join(ROOT, "data/validation/final_model_selection/selection.json"))})

    @app.get("/api/models")
    def models():
        out = {}
        for name in ("elliptic", "zenodo"):
            ms = sorted(glob.glob(os.path.join(ROOT, "artifacts", "models", name, "*", "manifest.json")))
            if ms:
                m = json.load(open(ms[-1]))
                out[name] = {k: m.get(k) for k in ("model_id", "model_version", "config", "score_type", "score_semantics",
                                                   "decision_threshold", "threshold", "requires_graph",
                                                   "feature_version", "created_utc")}
        return _clean(out)

    @app.post("/api/score")
    def score(events: List[dict] = Body(..., max_length=5000)):
        """Unified scoring interface over HTTP (same routing/validation as the stream consumer)."""
        if state["service"] is None:
            from src.scoring.service import ScoringService
            state["service"] = ScoringService()
        return [r.model_dump(mode="json") for r in state["service"].score_events(events)]

    return app


app = create_app()
