"""
Evaluation metrics, experiment logger, and report generation utilities.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any


def save_metrics_report(metrics: Dict[str, Any], output_path: str) -> None:
    """Save experiment metrics to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics saved to: {output_path}")


def format_streaming_table(metrics: Dict[str, Any]) -> str:
    """Format streaming benchmark metrics into GitHub markdown table."""
    return f"""| Metric | Value |
|---|---|
| **Raw Records Ingested** | {metrics.get('raw_records_read', 0):,} |
| **Completed Transactions** | {metrics.get('completed_transactions', 0):,} |
| **Micro-Batch Size** | {metrics.get('batch_size', 0)} |
| **Throughput (Transactions/sec)** | **{metrics.get('throughput_transactions_per_sec', 0):,} tx/s** |
| **Throughput (Raw Records/sec)** | **{metrics.get('throughput_raw_records_per_sec', 0):,} rec/s** |
| **Micro-Batch Latency p50** | **{metrics.get('latency_ms_p50', 0):.3f} ms** |
| **Micro-Batch Latency p95** | **{metrics.get('latency_ms_p95', 0):.3f} ms** |
| **Micro-Batch Latency p99** | **{metrics.get('latency_ms_p99', 0):.3f} ms** |
| **Micro-Batch Latency Mean** | {metrics.get('latency_ms_mean', 0):.3f} ms |
| **Graph Address Nodes Tracked** | {metrics.get('graph_addresses_tracked', 0):,} |
| **Graph Transaction Nodes** | {metrics.get('graph_transactions_tracked', 0):,} |
| **Memory Delta (RSS)** | {metrics.get('memory_delta_mb', 0)} MB (Peak: {metrics.get('memory_end_mb', 0)} MB) |
"""


# --------------------------------------------------------------------------- classifier metrics
# Added for notebooks 02/03. Illicit (1) is the positive class throughout.
import numpy as np  # noqa: E402

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.", "precision": "Precision (illicit)",
    "recall": "Recall (illicit)", "f1": "F1 (illicit)", "f0_5": "F0.5 (illicit)", "f2": "F2 (illicit)",
    "specificity": "Specificity", "fpr": "FPR", "fnr": "FNR", "mcc": "MCC", "kappa": "Cohen's kappa",
    "f1_macro": "F1 macro", "f1_weighted": "F1 weighted", "precision_macro": "Precision macro",
    "recall_macro": "Recall macro", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC (AP)", "log_loss": "Log-loss",
    "brier": "Brier", "ece": "ECE (15 bins)",
}
# Direction used for ranking: +1 higher is better, -1 lower is better.
METRIC_DIRECTION = {k: 1 for k in METRIC_LABELS}
METRIC_DIRECTION.update({"fpr": -1, "fnr": -1, "log_loss": -1, "brier": -1, "ece": -1})


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    """ECE with equal-width bins on the predicted illicit probability."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def _fbeta(prec, rec, beta):
    b2 = beta * beta
    return 0.0 if prec + rec == 0 else (1 + b2) * prec * rec / (b2 * prec + rec)


def full_metrics(y, p, threshold: float = 0.5) -> dict:
    """Every threshold-based and threshold-free metric of the report, from labels and P(illicit)."""
    from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score)
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = _fbeta(prec, rec, 1.0)
    f1_neg = _fbeta(npv, spec, 1.0)
    pos, neg = tp + fn, tn + fp
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    po = (tp + tn) / n
    pe = ((tp + fp) * pos + (tn + fn) * neg) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0
    both = len(np.unique(y)) == 2
    return {
        "accuracy": po, "balanced_accuracy": (rec + spec) / 2, "precision": prec, "recall": rec, "f1": f1,
        "f0_5": _fbeta(prec, rec, 0.5), "f2": _fbeta(prec, rec, 2.0), "specificity": spec,
        "fpr": 1 - spec, "fnr": 1 - rec, "mcc": float(mcc), "kappa": float(kappa),
        "f1_macro": (f1 + f1_neg) / 2, "f1_weighted": (f1 * pos + f1_neg * neg) / n,
        "precision_macro": (prec + npv) / 2, "recall_macro": (rec + spec) / 2,
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": float(threshold),
    }


def fast_counts_metrics(y, pred) -> dict:
    """F1/precision/recall/MCC from hard predictions (vectorised helper for bootstraps)."""
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)}


def per_timestep(y, p, steps, threshold: float = 0.5) -> "list[dict]":
    """Illicit precision / recall / F1 for each test time step (NaN-free: 0 when undefined)."""
    rows = []
    for t in np.unique(steps):
        m = steps == t
        pred = (np.asarray(p)[m] >= threshold).astype(int)
        r = fast_counts_metrics(np.asarray(y)[m], pred)
        r.update(time_step=int(t), n=int(m.sum()), n_illicit=int((np.asarray(y)[m] == 1).sum()))
        rows.append(r)
    return rows


def bootstrap_ci_seed_mean(y, probs_by_seed, metric_fn, n_boot: int = 1000, seed: int = 0,
                           alpha: float = 0.05):
    """95 % CI of the seed-averaged metric under resampling of test transactions.

    Each replicate draws test transactions with replacement, evaluates ``metric_fn(y, p)`` for
    every seed's predictions on that resample and averages over seeds. The interval therefore
    covers test-set sampling uncertainty of the 5-seed mean that the tables report.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = np.mean([metric_fn(y[idx], p[idx]) for p in probs_by_seed])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def bootstrap_metrics_table(y, probs_by_seed, n_boot: int = 1000, seed: int = 0, threshold: float = 0.5):
    """Mean +/- std over seeds and a 95 % bootstrap CI of the seed mean for every metric of full_metrics.

    One replicate = one resample of test transactions, scored for all seeds (see bootstrap_ci_seed_mean).
    Returns a DataFrame indexed by metric with columns mean, std, ci_low, ci_high.
    """
    import pandas as pd
    y = np.asarray(y)
    keys = list(METRIC_LABELS)
    per_seed = pd.DataFrame([full_metrics(y, p, threshold) for p in probs_by_seed])[keys]
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, len(keys)))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        reps[b] = np.mean([[full_metrics(y[idx], p[idx], threshold)[k] for k in keys] for p in probs_by_seed], axis=0)
    return pd.DataFrame({"mean": per_seed.mean(), "std": per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0,
                         "ci_low": np.nanquantile(reps, 0.025, axis=0), "ci_high": np.nanquantile(reps, 0.975, axis=0)},
                        index=keys)


REGRESSION_LABELS = {"RMSE": "RMSE", "MSE": "MSE", "MAE": "MAE", "MAPE_pct": "MAPE (%)", "R2": "R²",
                     "NRMSE": "NRMSE", "MedAE": "MedAE", "ME_bias": "Bias / ME"}


def probability_error_metrics(y, p) -> dict:
    """Regression-style error of the predicted P(illicit) against the true label (1 illicit, 0 licit).

    e = p - y. MSE equals the Brier score. MAPE is computed over illicit rows only (y = 1): a percentage
    error does not exist where the actual value is 0. NRMSE = RMSE / (max(y) - min(y)) = RMSE for 0/1 labels.
    R^2 = 1 - SSE / SST against the label mean. Bias > 0 means probabilities are too high on average.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    e = p - y
    mse = float((e ** 2).mean())
    sst = float(((y - y.mean()) ** 2).sum())
    pos = y == 1
    return {"RMSE": mse ** 0.5, "MSE": mse, "MAE": float(np.abs(e).mean()),
            "MAPE_pct": float(100 * np.mean(np.abs(e[pos]) / y[pos])) if pos.any() else float("nan"),
            "R2": 1 - float((e ** 2).sum()) / sst if sst > 0 else float("nan"),
            "NRMSE": mse ** 0.5 / (y.max() - y.min()) if y.max() > y.min() else float("nan"),
            "MedAE": float(np.median(np.abs(e))), "ME_bias": float(e.mean())}
