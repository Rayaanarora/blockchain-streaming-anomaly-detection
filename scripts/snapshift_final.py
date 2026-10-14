"""
SnapShift-Boost final evaluation (one-time test read).

Configuration fixed by scripts/snapshift_dev.py (pre-registered rule, steps 1-34 only): components "P".
Protocol identical to the baseline: train = labelled steps 1-29, threshold = F1-optimal on labelled
steps 30-34, test = labelled steps 35-49 evaluated once with that threshold.

Comparison: the baseline hgb_165_nbr (bundle 1.0.1) seed-42 probabilities stored by the final model
selection. Paired bootstrap (transactions and time steps) of the test F1 difference.

    .venv\\Scripts\\python.exe scripts\\snapshift_final.py --seed 42
Outputs: data/validation/snapshift/final/
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.elliptic_training import (assert_no_cross_step_edges, evaluate, load_spark_output,  # noqa: E402
                                          tune_threshold)
from src.models import snapshift  # noqa: E402

OUT = os.path.join(ROOT, "data/validation/snapshift/final")
COMPONENTS = "P"
BASELINE = "hgb_165_nbr"


def per_step(y, p, thr, steps):
    rows = []
    for s in np.unique(steps):
        m = steps == s
        pred = (p[m] >= thr).astype(int)
        rows.append({"time_step": int(s), "n": int(m.sum()), "illicit": int(y[m].sum()),
                     "recall": float(recall_score(y[m], pred, zero_division=0)),
                     "f1": float(f1_score(y[m], pred, zero_division=0))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    data = load_spark_output(os.path.join(ROOT, "data/processed/elliptic_spark"))
    assert_no_cross_step_edges(data)
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    Xs = StandardScaler().fit(data.X_raw[tr]).transform(data.X_raw).astype(np.float32)
    probs_path = os.path.join(OUT, f"seed_{a.seed}_probs.parquet")
    y = data.y
    if os.path.exists(probs_path):  # deterministic fit already done; reuse its probabilities
        saved = pd.read_parquet(probs_path).set_index("tx_id")
        p = np.full(len(y), np.nan)
        p[te | va] = saved.loc[data.tx_ids[te | va], "prob"].to_numpy()
    else:
        p = snapshift.fit_predict(data, Xs, tr, seed=a.seed, components=COMPONENTS)["p"]
    thr, val_f1 = tune_threshold(y[va], p[va])          # threshold from validation only
    test = evaluate(y[te], p[te], thr)                  # the single test read
    fit_s = time.time() - t0

    steps_te = data.time_steps[te]
    pre, post = steps_te <= 42, steps_te >= 43
    pred_te = (p[te] >= thr).astype(int)
    res = {"model": f"snapshift_{COMPONENTS}", "seed": a.seed, "components": COMPONENTS,
           "threshold": thr, "val_f1": val_f1, "test": test,
           "recall_steps_35_42": float(recall_score(y[te][pre], pred_te[pre])),
           "recall_steps_43_49": float(recall_score(y[te][post], pred_te[post])),
           "per_step": per_step(y[te], p[te], thr, steps_te), "seconds": round(fit_s, 1)}

    if not os.path.exists(probs_path):
        pd.DataFrame({"tx_id": data.tx_ids[te | va], "time_step": data.time_steps[te | va], "y_true": y[te | va],
                  "split": np.where(te[te | va], "test", "val"), "prob": p[te | va]}).to_parquet(
        probs_path, index=False)

    # ---- paired comparison with the baseline (same transactions)
    bp = pd.read_parquet(os.path.join(ROOT, "data/validation/final_model_selection/seed_42_probs.parquet"))
    bp = bp[(bp.config == BASELINE) & (bp.split == "test")].set_index("tx_id")
    b_thr = json.load(open(os.path.join(ROOT, "artifacts/models/elliptic/1.0.1/manifest.json")))["decision_threshold"]
    ids = data.tx_ids[te]
    b_prob = bp.loc[ids, "prob"].to_numpy()
    assert (bp.loc[ids, "y_true"].to_numpy() == y[te]).all()
    b_pred = (b_prob >= b_thr).astype(int)
    yt = y[te]
    rng = np.random.default_rng(0)
    diff_tx, diff_step = [], []
    uniq = np.unique(steps_te)
    by_step = {s: np.flatnonzero(steps_te == s) for s in uniq}
    for _ in range(a.bootstrap):
        i = rng.integers(0, len(yt), len(yt))
        diff_tx.append(f1_score(yt[i], pred_te[i]) - f1_score(yt[i], b_pred[i]))
        j = np.concatenate([by_step[s] for s in rng.choice(uniq, len(uniq))])
        diff_step.append(f1_score(yt[j], pred_te[j], zero_division=0) - f1_score(yt[j], b_pred[j], zero_division=0))
    b_eval = evaluate(yt, b_prob, b_thr)
    res["baseline"] = {"config": BASELINE, "threshold": b_thr, "test": b_eval,
                       "recall_steps_35_42": float(recall_score(yt[pre], b_pred[pre])),
                       "recall_steps_43_49": float(recall_score(yt[post], b_pred[post]))}
    res["f1_difference"] = {
        "point": test["f1_illicit"] - b_eval["f1_illicit"],
        "ci95_bootstrap_transactions": [float(np.percentile(diff_tx, 2.5)), float(np.percentile(diff_tx, 97.5))],
        "ci95_bootstrap_time_steps": [float(np.percentile(diff_step, 2.5)), float(np.percentile(diff_step, 97.5))],
        "share_bootstrap_transactions_positive": float(np.mean(np.array(diff_tx) > 0)),
        "share_bootstrap_time_steps_positive": float(np.mean(np.array(diff_step) > 0)),
        "n_bootstrap": a.bootstrap}
    json.dump(res, open(os.path.join(OUT, f"seed_{a.seed}.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_step"}, indent=2))


if __name__ == "__main__":
    main()
