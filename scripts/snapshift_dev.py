"""
SnapShift-Boost development: choose components using steps 1-34 only (test steps 35-49 are removed
from the data before anything else happens).

Rolling-origin folds (train -> validate):
  fold 1: steps 1-19 -> 20-24      fold 2: steps 1-24 -> 25-29      fold 3: steps 1-29 -> 30-34

Variants: every subset of the components R (snapshot ranks), I (snapshot anomaly score),
P (stage-2 score propagation); "B" = no components = the current baseline hgb_165_nbr.

Pre-registered selection rule (fixed before the first run):
  1. primary metric: mean over the three folds of validation F1 at that fold's F1-optimal threshold
  2. a variant replaces the baseline only if its mean exceeds the baseline's by >= 0.005;
     among variants within 0.005 of the best, take the one with fewer components, then higher mean AP
HGB without early stopping is deterministic, so one seed is used here.

    .venv\\Scripts\\python.exe scripts\\snapshift_dev.py --fold 1     # one fold per process (parallel)
    .venv\\Scripts\\python.exe scripts\\snapshift_dev.py --summarise
Outputs: data/validation/snapshift/dev/
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "5")

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.elliptic_training import (EllipticData, assert_no_cross_step_edges, evaluate,  # noqa: E402
                                          load_spark_output, tune_threshold)
from src.models import snapshift  # noqa: E402

OUT = os.path.join(ROOT, "data/validation/snapshift/dev")
FOLDS = {1: ((1, 19), (20, 24)), 2: ((1, 24), (25, 29)), 3: ((1, 29), (30, 34))}
VARIANTS = ["B", "R", "I", "P", "RI", "RP", "IP", "RIP"]


def dev_data() -> EllipticData:
    d = load_spark_output(os.path.join(ROOT, "data/processed/elliptic_spark"))
    assert_no_cross_step_edges(d)
    keep = d.time_steps <= 34
    remap = -np.ones(len(keep), dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    e = d.edge_index[:, keep[d.edge_index[0]] & keep[d.edge_index[1]]]
    return EllipticData(d.tx_ids[keep], d.time_steps[keep], d.X_raw[keep], d.y[keep], remap[e])


def run_fold(k: int):
    data = dev_data()
    assert data.time_steps.max() <= 34
    (a, b), (c, e) = FOLDS[k]
    lab = data.y >= 0
    tr = lab & (data.time_steps >= a) & (data.time_steps <= b)
    va = lab & (data.time_steps >= c) & (data.time_steps <= e)
    Xs = StandardScaler().fit(data.X_raw[tr]).transform(data.X_raw).astype(np.float32)
    cache, res = {}, {}
    path = os.path.join(OUT, f"fold_{k}.json")
    if os.path.exists(path):
        res = json.load(open(path))["variants"]
    for v in VARIANTS:
        if v in res:
            continue
        t0 = time.time()
        out = snapshift.fit_predict(data, Xs, tr, seed=42, components="" if v == "B" else v, cache=cache)
        p = out["p"][va]
        thr, f1 = tune_threshold(data.y[va], p)
        m = evaluate(data.y[va], p, thr)
        res[v] = {"val_f1_at_own_threshold": f1, "val_ap": float(average_precision_score(data.y[va], p)),
                  "threshold": thr, "precision": m["precision"], "recall": m["recall"],
                  "f1_at_0.5": evaluate(data.y[va], p, 0.5)["f1_illicit"], "seconds": round(time.time() - t0, 1)}
        print(time.strftime("%H:%M:%S"), f"fold {k} {v}", json.dumps(res[v]), flush=True)
        json.dump({"fold": k, "train_steps": FOLDS[k][0], "val_steps": FOLDS[k][1],
                   "n_train": int(tr.sum()), "n_val": int(va.sum()), "variants": res}, open(path, "w"), indent=2)


def summarise():
    folds = [json.load(open(os.path.join(OUT, f"fold_{k}.json"))) for k in FOLDS]
    rows = {}
    for v in VARIANTS:
        f1s = [f["variants"][v]["val_f1_at_own_threshold"] for f in folds]
        aps = [f["variants"][v]["val_ap"] for f in folds]
        rows[v] = {"mean_f1": float(np.mean(f1s)), "mean_ap": float(np.mean(aps)), "fold_f1": f1s, "fold_ap": aps}
    base = rows["B"]["mean_f1"]
    best = max(rows[v]["mean_f1"] for v in VARIANTS)
    near = [v for v in VARIANTS if v != "B" and rows[v]["mean_f1"] >= best - 0.005]
    near.sort(key=lambda v: (len(v), -rows[v]["mean_ap"]))
    choice = near[0] if near and rows[near[0]]["mean_f1"] >= base + 0.005 else "B"
    summary = {"rule": __doc__.split("Pre-registered selection rule")[1].split("HGB without")[0].strip(),
               "variants": rows, "selected": choice}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    for v in VARIANTS:
        r = rows[v]
        print(f"{v:4s} mean F1 {r['mean_f1']:.4f}  mean AP {r['mean_ap']:.4f}  folds "
              + " ".join(f"{x:.4f}" for x in r["fold_f1"]))
    print("selected:", choice)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, choices=list(FOLDS))
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.summarise:
        summarise()
    else:
        run_fold(a.fold)
