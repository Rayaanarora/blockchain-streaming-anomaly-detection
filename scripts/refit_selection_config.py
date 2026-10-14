"""
Refit ONE configuration of the final model selection after a code fix, without rerunning the
(multi-hour) GAT parts, and splice its results into the existing per-seed outputs.

Why this exists: neighbor_aggregate_features was made order-invariant (see its docstring). Only
the configuration that uses those features (hgb_165_nbr) is affected; every other configuration's
records are left untouched.

Same protocol as scripts/run_final_model_selection.py: same data source, train-only scaler,
grid-selected HistGBDT parameters from grid_search.json, F1-maximising threshold on validation,
test evaluated once with that threshold.

Before writing, the original seed_<s>.json / seed_<s>_probs.parquet / fitted_seed42/models.joblib
are copied to <out>/backup_before_refit_<config>_<timestamp>/.

Usage:
    .venv\\Scripts\\python.exe scripts\\refit_selection_config.py --config hgb_165_nbr
    .venv\\Scripts\\python.exe scripts\\run_final_model_selection.py --aggregate-only --model-version 1.0.1
"""
import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from scripts.run_final_model_selection import SEEDS, fit_predict, hgb  # noqa: E402
from src.models.elliptic_training import (evaluate, load_spark_output, neighbor_aggregate_features,  # noqa: E402
                                          tune_threshold)

SUPPORTED = {"hgb_165_nbr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(SUPPORTED))
    ap.add_argument("--spark-dir", default=os.path.join(ROOT, "data/processed/elliptic_spark"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/validation/final_model_selection"))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731

    data = load_spark_output(args.spark_dir)
    tr, va, te, y = data.train_mask, data.val_mask, data.test_mask, data.y
    ratio = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    X = StandardScaler().fit(data.X_raw[tr]).transform(data.X_raw).astype(np.float32)
    NBR = neighbor_aggregate_features(X[:, :93], data.edge_index, len(X))
    Xa = np.hstack([X, NBR])
    grid = json.load(open(os.path.join(args.out, "grid_search.json")))
    best_hgb = max(grid["hgb"], key=lambda r: r["val_ap"])["params"]

    backup = os.path.join(args.out, f"backup_before_refit_{args.config}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(backup)
    fitted_dir = os.path.join(args.out, "fitted_seed42")
    for s in args.seeds:
        for f in (f"seed_{s}.json", f"seed_{s}_probs.parquet"):
            shutil.copy2(os.path.join(args.out, f), backup)
    shutil.copy2(os.path.join(fitted_dir, "models.joblib"), backup)
    log("backed up originals to", backup)

    for seed in args.seeds:
        m, (pv, pt), fit_s = fit_predict(hgb(best_hgb, seed, ratio), Xa[tr], y[tr], [Xa[va], Xa[te]])
        th, vf1 = tune_threshold(y[va], pv)
        rec = {"seed": seed, "config": args.config, "val_f1": vf1,
               "val_ap": float(average_precision_score(y[va], pv)), "fit_seconds": fit_s,
               **evaluate(y[te], pt, th)}
        path = os.path.join(args.out, f"seed_{seed}.json")
        recs = [r for r in json.load(open(path)) if r["config"] != args.config] + [rec]
        json.dump(recs, open(path, "w"), indent=1)
        ppath = os.path.join(args.out, f"seed_{seed}_probs.parquet")
        probs = pd.read_parquet(ppath)
        probs = probs[probs.config != args.config]
        new = [pd.DataFrame({"seed": seed, "config": args.config, "split": "test", "tx_id": data.tx_ids[te],
                             "time_step": data.time_steps[te], "y_true": y[te], "prob": pt}),
               pd.DataFrame({"seed": seed, "config": args.config, "split": "val", "tx_id": data.tx_ids[va],
                             "time_step": data.time_steps[va], "y_true": y[va], "prob": pv})]
        pd.concat([probs] + new).to_parquet(ppath, index=False)
        if seed == 42:
            models = joblib.load(os.path.join(fitted_dir, "models.joblib"))
            models[args.config] = m
            joblib.dump(models, os.path.join(fitted_dir, "models.joblib"))
        log(f"seed {seed}: val_f1={vf1:.4f} val_ap={rec['val_ap']:.4f} threshold={th} (test f1, info only: "
            f"{rec['f1_illicit']:.4f})")


if __name__ == "__main__":
    main()
