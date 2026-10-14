"""
Independently recompute Elliptic test metrics from saved prediction files and
compare them with the metrics that were reported alongside them.

Checks performed (Phase 1 audit):
  * positive class is "illicit" (y_true == 1) and labels are binary (no -1 rows)
  * every model in a prediction file is evaluated on the identical row set
  * test rows are exactly the labelled nodes of steps 35..49 in the preprocessed npz
  * confusion matrices recomputed at the reported validation thresholds reproduce
    the reported precision / recall / F1 (saved probabilities are rounded to 4 dp,
    so tiny deviations are reported rather than hidden)
  * the 5-seed CSV really contains 5 seeds x N configurations

Usage:
    .venv\\Scripts\\python.exe scripts\\verify_saved_metrics.py
Writes: data/validation/metric_verification/metric_verification.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NPZ = os.path.join(ROOT, "data/processed/elliptic/elliptic_preprocessed.npz")
IMPROVED = os.path.join(ROOT, "data/validation/tgat_boostnet_improved")
OUT_DIR = os.path.join(ROOT, "data/validation/metric_verification")

# prediction column -> config key in metrics.json["seed_42_single_run"]
IMPROVED_COLUMNS = {
    "prob_hist_165": "1_HistGBDT_165",
    "prob_xgb_165": "2_XGBoost_Baseline_165",
    "prob_tgat_alone": "3_TGAT_Alone_32",
    "prob_tgat_boostnet_base": "4_TGAT_BoostNet_Baseline_197",
    "prob_tgat_histgbdt": "5_TGAT_HistGBDT_197",
    "prob_tgat_boostnet_opt": "7_TGAT_BoostNet_Optimized_197",
    "prob_hybrid_ensemble": "8_TGAT_Hybrid_Ensemble",
}


def metrics_at(y, p, th):
    pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_illicit": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"checks": {}, "models": {}, "issues": []}

    data = np.load(NPZ)
    ts, y_all, tx = data["time_steps"], data["y"], data["tx_ids"]
    test_mask = (ts >= 35) & (y_all != -1)

    preds = pd.read_parquet(os.path.join(IMPROVED, "test_predictions.parquet"))
    metrics = json.load(open(os.path.join(IMPROVED, "metrics.json")))
    seeds = pd.read_csv(os.path.join(IMPROVED, "seed_results.csv"))

    y = preds["y_true"].to_numpy()
    report["checks"]["labels_binary"] = bool(set(np.unique(y)) <= {0, 1})
    report["checks"]["n_rows"] = int(len(preds))
    report["checks"]["n_illicit"] = int((y == 1).sum())
    report["checks"]["test_rows_equal_npz_test_mask"] = bool(
        np.array_equal(preds["tx_id"].to_numpy(), tx[test_mask])
        and np.array_equal(y, y_all[test_mask])
    )
    report["checks"]["time_steps_35_to_49"] = bool(preds["time_step"].between(35, 49).all())
    report["checks"]["same_rows_for_all_models"] = True  # single file, single row set

    counts = seeds.groupby("config")["seed"].apply(lambda s: sorted(s.tolist())).to_dict()
    report["checks"]["seed_runs_per_config"] = counts
    report["checks"]["five_seeds_each"] = all(v == [42, 123, 456, 789, 2024] for v in counts.values())

    for col, cfg in IMPROVED_COLUMNS.items():
        rep = metrics["seed_42_single_run"][cfg]
        th = float(rep["selected_threshold"])
        got = metrics_at(y, preds[col].to_numpy(dtype=np.float64), th)
        exp = rep["metrics"]
        diffs = {k: round(got[k] - exp[k], 4) for k in ("precision", "recall", "f1_illicit", "roc_auc", "average_precision")}
        cm_match = got["confusion_matrix"] == exp["confusion_matrix"]
        # internal consistency of the *reported* confusion matrix vs reported P/R/F1
        c = exp["confusion_matrix"]
        p_cm = c["tp"] / max(c["tp"] + c["fp"], 1)
        r_cm = c["tp"] / max(c["tp"] + c["fn"], 1)
        f_cm = 2 * p_cm * r_cm / max(p_cm + r_cm, 1e-12)
        consistent = abs(p_cm - exp["precision"]) < 1e-4 and abs(r_cm - exp["recall"]) < 1e-4 and abs(f_cm - exp["f1_illicit"]) < 1e-4
        report["models"][cfg] = {
            "column": col, "threshold": th, "recomputed": got, "reported": exp,
            "abs_diff": diffs, "confusion_matrix_match": cm_match,
            "reported_cm_consistent_with_reported_prf": consistent,
        }
        if not cm_match or max(abs(v) for v in diffs.values()) > 5e-4:
            report["issues"].append(f"{cfg}: recomputed metrics differ from reported (diff={diffs}, cm_match={cm_match}); "
                                    "saved probabilities are rounded to 4 decimals")

    report["issues"].append("test_predictions.parquet only contains seed 42; seeds 123/456/789/2024 cannot be "
                            "re-verified from predictions (only their summary rows exist in seed_results.csv)")
    report["issues"].append("6_Tuned_XGBoost_165 probabilities were not saved, so its seed-42 metrics cannot be recomputed")
    xgb_std = seeds[seeds.config == "2_XGBoost_Baseline_165"]["f1_illicit"].std()
    if xgb_std == 0:
        report["issues"].append("2_XGBoost_Baseline_165 has zero variance across seeds: no row/column subsampling, so "
                                "the seed has no effect; its '5 seeds' are 5 identical runs")

    out = os.path.join(OUT_DIR, "metric_verification.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["checks"], indent=1))
    for cfg, m in report["models"].items():
        print(f"{cfg:32s} cm_match={m['confusion_matrix_match']} diff={m['abs_diff']} reported_consistent={m['reported_cm_consistent_with_reported_prf']}")
    print("issues:")
    for i in report["issues"]:
        print(" -", i)
    print("written", out)
    return 0 if all(m["reported_cm_consistent_with_reported_prf"] for m in report["models"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
