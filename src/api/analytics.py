"""
Read-only analytics behind the dashboard's charts. Everything is derived from files the pipeline wrote:

  stream_analytics(run)   predictions/*.parquet of one streaming run, plus the Elliptic ground-truth
                          labels (data/raw/elliptic/elliptic_txs_classes.csv). Labels are joined here, at
                          display time, purely to evaluate what was streamed; the scorer never sees them.
  ego_graph(run, tx)      the transaction's same-snapshot neighbourhood from the Spark edge list
                          (data/processed/elliptic_spark/edges), coloured by the run's scores.
  offline_results()       tables/*.csv written by notebooks 01-04 and the step-43 audit JSON.
"""
import functools
import glob
import json
import os
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LABELS_CSV = os.path.join(ROOT, "data", "raw", "elliptic", "elliptic_txs_classes.csv")
EDGES_DIR = os.path.join(ROOT, "data", "processed", "elliptic_spark", "edges")
TABLES = os.path.join(ROOT, "tables")
STEP43 = os.path.join(ROOT, "data", "validation", "final_model_selection", "step43_shift")
REASON_RE = re.compile(r"^(?P<feat>[^:]+):(?P<dir>high|low)\((?P<z>[+-]?[0-9.]+)\)$")
SCORE_BINS = 40


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@functools.lru_cache(maxsize=1)
def elliptic_labels() -> Optional[pd.Series]:
    """tx_id (str) -> 1 illicit, 0 licit, -1 unknown. None if the raw CSV is absent."""
    if not os.path.exists(LABELS_CSV):
        return None
    c = pd.read_csv(LABELS_CSV, dtype={"txId": str, "class": str})
    return c.set_index("txId")["class"].map({"1": 1, "2": 0}).fillna(-1).astype(int)


def _counts(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp = int(((y == 1) & flagged).sum())
    fp = int(((y == 0) & flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    tn = int(((y == 0) & ~flagged).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def _downsample(n: int, k: int) -> np.ndarray:
    return np.arange(n) if n <= k else np.unique(np.linspace(0, n - 1, k).astype(int))


def stream_analytics(pred: pd.DataFrame) -> dict:
    out = {"n": int(len(pred))}
    if not len(pred):
        return out
    p = pred.copy()
    p["ts"] = pd.to_datetime(p["scored_at"], utc=True, errors="coerce")
    t0 = p["ts"].min()
    p["t"] = (p["ts"] - t0).dt.total_seconds()
    out["t0"] = t0.isoformat() if pd.notna(t0) else None

    # ---------------------------------------------------------------- event flow over wall time
    flow = {}
    for ds, g in p.sort_values("t").groupby("source_dataset"):
        t = g["t"].to_numpy(float)
        cum = np.arange(1, len(t) + 1)
        fl = np.cumsum(g["flagged"].fillna(False).astype(bool).to_numpy())
        i = _downsample(len(t), 240)
        flow[ds] = {"t": t[i].round(3).tolist(), "scored": cum[i].tolist(), "flagged": fl[i].tolist()}
    out["flow"] = flow
    fl = p[p["flagged"].fillna(False).astype(bool)].sort_values("t")
    i = _downsample(len(fl), 600)
    out["flag_events"] = [{"t": round(float(r.t), 3), "ds": r.source_dataset, "score": float(r.score),
                           "tx": str(r.transaction_id)} for r in fl.iloc[i].itertuples()]

    # ---------------------------------------------------------------- Elliptic: labelled evaluation per snapshot
    e = p[(p["source_dataset"] == "elliptic") & (p["status"] == "scored")]
    if len(e):
        labels = elliptic_labels()
        e = e.assign(step=e["event_time_value"].astype(int))
        y = e["transaction_id"].astype(str).map(labels).fillna(-1).astype(int).to_numpy() if labels is not None \
            else -np.ones(len(e), dtype=int)
        e = e.assign(label=y)
        thr = float(e["threshold"].iloc[0])
        edges = np.linspace(0, 1, SCORE_BINS + 1)
        steps = []
        for step, g in e.groupby("step"):
            s = g["score"].to_numpy(float)
            lab = g["label"].to_numpy()
            flg = g["flagged"].fillna(False).astype(bool).to_numpy()
            known = lab >= 0
            steps.append({
                "step": int(step), "n": int(len(g)), "labelled": int(known.sum()),
                "illicit": int((lab == 1).sum()), "flagged": int(flg.sum()),
                "hist": {name: np.histogram(s[lab == v], bins=edges)[0].tolist()
                         for name, v in (("illicit", 1), ("licit", 0), ("unknown", -1))},
                "model_version": str(g["model_version"].iloc[0]),
                **_counts(lab[known], flg[known])})
        known = e["label"].to_numpy() >= 0
        lab_all = e["label"].to_numpy()[known]
        sc_all = e["score"].to_numpy(float)[known]
        st_all = e["step"].to_numpy()[known]
        out["elliptic"] = {
            "threshold": thr, "bin_edges": edges.round(4).tolist(), "steps": steps,
            "model_version": str(e["model_version"].iloc[-1]), "labels_available": labels is not None,
            "overall": _counts(lab_all, sc_all >= thr),
            # compact arrays for the client-side threshold explorer (labelled rows only)
            "labelled": {"score": np.round(sc_all, 5).tolist(), "label": lab_all.tolist(), "step": st_all.tolist()},
            "label_note": "Ground-truth labels joined at display time from elliptic_txs_classes.csv; "
                          "the scorer never receives them. Unknown-label rows are excluded from metrics."}

    # ---------------------------------------------------------------- Zenodo: score distribution + reason codes
    z = p[(p["source_dataset"] == "zenodo") & (p["status"] == "scored")]
    if len(z):
        s = np.sort(z["score"].to_numpy(float))
        q = np.linspace(0, 1, 201)
        thr = float(z["threshold"].iloc[0])
        feats = defaultdict(lambda: {"count": 0, "rank1": 0, "z": []})
        q90 = float(np.quantile(s, 0.9))
        for rc in z.loc[z["score"] >= q90, "reason_codes"].fillna(""):
            for rank, part in enumerate([x for x in rc.split(";") if x]):
                m = REASON_RE.match(part.strip())
                if not m:
                    continue
                f = feats[f"{m['feat']} ({m['dir']})"]
                f["count"] += 1
                f["rank1"] += rank == 0
                f["z"].append(float(m["z"]))
        out["zenodo"] = {
            "threshold": thr, "n": int(len(s)), "flagged": int((s >= thr).sum()),
            "ecdf": {"score": np.quantile(s, q).round(5).tolist(), "p": q.round(4).tolist()},
            "reasons_scope": f"top decile of scores (score >= {q90:.3f}, n = {int((s >= q90).sum())})",
            "tail": np.round(s[-min(len(s), 300):], 5).tolist(),
            "hist": dict(zip(("counts", "edges"), [a.tolist() for a in np.histogram(s, bins=40)])),
            "reasons": sorted(({"feature": k, "count": v["count"], "rank1": v["rank1"],
                                "z_median": float(np.median(v["z"])), "z": v["z"][:200]}
                               for k, v in feats.items()), key=lambda r: -r["count"])}

    # ---------------------------------------------------------------- per-batch rows and statuses
    b = p.groupby(["batch_id", "source_dataset"]).size().unstack(fill_value=0)
    out["rows_per_batch"] = {"batch_id": b.index.astype(int).tolist(),
                             **{c: b[c].astype(int).tolist() for c in b.columns}}
    out["status_counts"] = p["status"].value_counts().to_dict()
    return _clean(out)


@functools.lru_cache(maxsize=64)
def _edges_for_step(step: int) -> pd.DataFrame:
    return pd.read_parquet(EDGES_DIR, columns=["source_tx_id", "target_tx_id", "time_step"],
                           filters=[("time_step", "==", step)])


def ego_graph(pred: pd.DataFrame, tx: str, max_nodes: int = 140) -> dict:
    row = pred[(pred["source_dataset"] == "elliptic") & (pred["transaction_id"].astype(str) == str(tx))]
    if not len(row):
        raise KeyError(tx)
    step = int(row["event_time_value"].iloc[0])
    if not os.path.isdir(EDGES_DIR):
        return {"available": False, "reason": "Spark edge list (data/processed/elliptic_spark/edges) not found"}
    ed = _edges_for_step(step)
    src = ed["source_tx_id"].astype(str).to_numpy()
    dst = ed["target_tx_id"].astype(str).to_numpy()
    nbr = defaultdict(set)
    for a, b in zip(src, dst):
        nbr[a].add(b)
        nbr[b].add(a)
    hop = {str(tx): 0}
    frontier = [str(tx)]
    for h in (1, 2):
        nxt = []
        for u in frontier:
            for v in sorted(nbr[u]):
                if v not in hop and len(hop) < max_nodes:
                    hop[v] = h
                    nxt.append(v)
        frontier = nxt
    ids = set(hop)
    snap = pred[(pred["source_dataset"] == "elliptic") & (pred["event_time_value"] == step)]
    sc = dict(zip(snap["transaction_id"].astype(str), snap["score"]))
    fg = dict(zip(snap["transaction_id"].astype(str), snap["flagged"].fillna(False)))
    labels = elliptic_labels()
    nodes = [{"id": n, "hop": h, "score": sc.get(n), "flagged": bool(fg.get(n, False)),
              "label": int(labels.get(n, -1)) if labels is not None else -1, "degree": len(nbr[n])}
             for n, h in hop.items()]
    links = [{"s": a, "t": b} for a, b in zip(src, dst) if a in ids and b in ids]
    return _clean({"available": True, "tx": str(tx), "step": step, "threshold": float(row["threshold"].iloc[0]),
                   "nodes": nodes, "links": links, "truncated": len(hop) >= max_nodes,
                   "snapshot_nodes": int(len(snap)), "snapshot_edges": int(len(ed))})


def _csv(name: str, **kw) -> Optional[pd.DataFrame]:
    path = os.path.join(TABLES, name)
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def _records(df: Optional[pd.DataFrame]):
    return [] if df is None else df.replace({np.nan: None}).to_dict("records")


def offline_results() -> dict:
    mean = _csv("03_leaderboard_all_metrics_mean.csv", index_col=0)
    std = _csv("03_leaderboard_all_metrics_std.csv", index_col=0)
    models = []
    if mean is not None:
        for name, r in mean.iterrows():
            m = {"model": name, "ours": "TGAT-BoostNet" in name}
            for k in ("f1", "precision", "recall", "pr_auc", "roc_auc", "mcc", "balanced_accuracy", "specificity"):
                m[k] = r.get(k)
                m[k + "_sd"] = std.loc[name, k] if std is not None and name in std.index and k in std.columns else None
            models.append(m)
    eff = _csv("03_efficiency.csv", index_col=0)
    eff_rows = [] if eff is None else [dict(model=i, **r) for i, r in eff.replace({np.nan: None}).iterrows()]
    sig = _csv("03_significance.csv", index_col=0)
    sig_rows = []
    if sig is not None:
        for name, r in sig.iterrows():
            lo, hi = [float(x) for x in str(r["ΔF1 95% CI"]).strip("[]").split(",")]
            sig_rows.append({"model": name, "d_f1": r["ΔF1"], "lo": lo, "hi": hi, "p_holm": r["p (F1 boot) Holm"],
                             "p_mcnemar_holm": r["p (McNemar) Holm"], "verdict": r["verdict"]})
    tuned = _csv("04_tuned_vs_untuned.csv", index_col=0)
    tuned_rows = []
    if tuned is not None:
        for name, r in tuned.iterrows():
            fam, kind = name.rsplit(" (", 1)
            tuned_rows.append({"family": fam, "kind": kind.rstrip(")"), "threshold": r["threshold"],
                               "f1": float(str(r["F1 (illicit)"]).split(" ± ")[0]),
                               "f1_sd": float(str(r["F1 (illicit)"]).split(" ± ")[1]),
                               "pr_auc": float(str(r["PR-AUC (AP)"]).split(" ± ")[0])})
    per_step = _csv("03_per_timestep_f1.csv")
    prev = None
    p_path = os.path.join(STEP43, "prevalence_per_step.csv")
    if os.path.exists(p_path):
        prev = pd.read_csv(p_path)
    s43 = json.load(open(os.path.join(STEP43, "step43_shift.json"))) if os.path.exists(
        os.path.join(STEP43, "step43_shift.json")) else None
    rvr = _csv("03_reproduced_vs_reported.csv")
    abl = _csv("03_ablation.csv", index_col=0)
    abl_rows = [] if abl is None else [{"variant": i, "f1": r["F1"], "f1_sd": r["F1 sd"], "pr_auc": r["PR-AUC"],
                                        "d_f1": r["ΔF1 vs full"]} for i, r in abl.iterrows()]
    return _clean({"models": models, "efficiency": eff_rows, "significance": sig_rows, "tuned": tuned_rows,
                   "per_step_f1": _records(per_step), "prevalence": _records(prev), "step43": s43,
                   "reproduced_vs_reported": _records(rvr), "ablation": abl_rows,
                   "sources": {"leaderboard": "tables/03_leaderboard_all_metrics_{mean,std}.csv",
                               "per_step": "tables/03_per_timestep_f1.csv",
                               "significance": "tables/03_significance.csv",
                               "tuned": "tables/04_tuned_vs_untuned.csv",
                               "efficiency": "tables/03_efficiency.csv",
                               "step43": "data/validation/final_model_selection/step43_shift/"}})
