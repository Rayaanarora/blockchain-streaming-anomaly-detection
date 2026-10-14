"""
Export SnapShift-Boost (components "P") as Elliptic bundle 2.0.0 and verify it.

Checks, all of which must pass before the bundle is written:
  1. the refit reproduces the probabilities evaluated by scripts/snapshift_final.py (val + test)
  2. a second seed gives identical probabilities (the model is deterministic)
After export:
  3. the bundle, scoring one complete time-step snapshot at a time (as the streaming consumer does),
     reproduces the evaluated test probabilities and metrics

    .venv\\Scripts\\python.exe scripts\\export_snapshift_bundle.py
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.elliptic_bundle import EllipticModelBundle, export_bundle  # noqa: E402
from src.models.elliptic_training import (assert_no_cross_step_edges, evaluate, load_spark_output,  # noqa: E402
                                          tune_threshold)
from src.models import snapshift  # noqa: E402

FINAL = os.path.join(ROOT, "data/validation/snapshift/final")
TOL = 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="2.0.0")
    ap.add_argument("--check-seed", type=int, default=123)
    a = ap.parse_args()
    log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)  # noqa: E731
    final = json.load(open(os.path.join(FINAL, "seed_42.json")))
    saved = pd.read_parquet(os.path.join(FINAL, "seed_42_probs.parquet")).set_index("tx_id")

    data = load_spark_output(os.path.join(ROOT, "data/processed/elliptic_spark"))
    assert_no_cross_step_edges(data)
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    scaler = StandardScaler().fit(data.X_raw[tr])
    Xs = scaler.transform(data.X_raw).astype(np.float32)
    ev = va | te
    ref = saved.loc[data.tx_ids[ev], "prob"].to_numpy()

    out = snapshift.fit_predict(data, Xs, tr, seed=42, components=final["components"])
    d42 = float(np.abs(out["p"][ev] - ref).max())
    log("check 1: refit vs evaluated probabilities, max |diff| =", d42)
    assert d42 <= TOL, "refit does not reproduce the evaluated model"
    other = snapshift.fit_predict(data, Xs, tr, seed=a.check_seed, components=final["components"])
    dseed = float(np.abs(other["p"] - out["p"]).max())
    log(f"check 2: seed {a.check_seed} vs seed 42, max |diff| over all nodes =", dseed)
    assert dseed <= TOL, "model is not deterministic across seeds"

    thr, val_f1 = tune_threshold(data.y[va], out["p"][va])
    assert thr == final["threshold"]
    m = out["models"]
    ratio = float((data.y[tr] == 0).sum() / (data.y[tr] == 1).sum())
    summary = {"config": "snapshift_P", "val_f1": val_f1, **{f"test_{k}": v for k, v in final["test"].items()},
               "recall_steps_35_42": final["recall_steps_35_42"], "recall_steps_43_49": final["recall_steps_43_49"],
               "baseline_comparison": {"baseline": final["baseline"]["config"],
                                       "baseline_test_f1": final["baseline"]["test"]["f1_illicit"],
                                       "f1_difference": final["f1_difference"]},
               "deterministic_across_seeds": True}
    meta = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "spark",
            "splits": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
            "class_weight_ratio": ratio, "hgb_params": snapshift.HGB_PARAMS,
            "stage1_cross_fit": "4 contiguous groups of training time steps (out-of-fold stage-1 scores for "
                                "training rows); full-training-set stage-1 model for every other row and at inference",
            "propagation_features": ["own stage-1 p", "in-neighbour mean", "in-neighbour max", "out-neighbour mean",
                                     "out-neighbour max", "undirected mean", "undirected max", "undirected 2-hop mean"],
            "selection": "data/validation/snapshift/dev/summary.json (rolling folds within steps 1-34, "
                         "pre-registered rule)",
            "evaluation": "data/validation/snapshift/final/seed_42.json (single test read)"}
    path = export_bundle(os.path.join(ROOT, "artifacts/models/elliptic"), a.version, "snapshift_P",
                         {"snapshift_stage1": m["stage1"], "snapshift_stage2": m["stage2"]}, scaler, None, thr,
                         final["test"], summary, meta)
    log("exported", path)

    # check 3: per-snapshot scoring through the bundle
    b = EllipticModelBundle(path)
    nodes_all = pd.DataFrame(data.X_raw, columns=[f"f{i}" for i in range(1, 166)])
    nodes_all.insert(0, "tx_id", data.tx_ids)
    src, dst = data.edge_index
    scores = np.empty(len(data.y))
    for s in range(35, 50):
        idx = np.flatnonzero(data.time_steps == s)
        e = (data.time_steps[src] == s)
        edges = pd.DataFrame({"source_tx_id": data.tx_ids[src[e]], "target_tx_id": data.tx_ids[dst[e]]})
        res = b.predict(nodes_all.iloc[idx].reset_index(drop=True), edges)
        assert (res.graph_context == "complete").all()
        scores[idx] = res["score"].to_numpy()
    d3 = float(np.abs(scores[te] - saved.loc[data.tx_ids[te], "prob"].to_numpy()).max())
    m3 = evaluate(data.y[te], scores[te], b.threshold)
    log("check 3: per-snapshot bundle scores vs evaluated, max |diff| =", d3, "test F1", round(m3["f1_illicit"], 4))
    assert d3 <= 1e-9 and abs(m3["f1_illicit"] - final["test"]["f1_illicit"]) < 1e-12
    json.dump({"bundle": path, "check1_refit_max_diff": d42, "check2_seed_max_diff": dseed,
               "check3_snapshot_max_diff": d3, "check3_test_metrics": m3},
              open(os.path.join(FINAL, "bundle_checks.json"), "w"), indent=2)
    log("all checks passed")


if __name__ == "__main__":
    main()
