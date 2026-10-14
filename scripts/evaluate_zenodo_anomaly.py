"""
Evaluation of the Zenodo anomaly detector WITHOUT ground-truth labels.

No supervised precision/recall/F1/ROC-AUC against "illicit" is reported: no reliable labels exist.
Instead:
  1. Score-distribution stability across block chunks (PSI vs the fit window, flag rate per chunk)
  2. Seed stability (refits with other seeds: Spearman rank correlation, top-0.1% Jaccard overlap)
  3. Sampling impact (fit-sample size 0.5% / 2% / 5%): same statistics vs the production model
  4. Feature sensitivity (drop feature groups, refit, compare rankings)
  5. SYNTHETIC fault injection on held-out real transactions (clearly separated from real-world claims)
  6. Documented inspection of the top-ranked real transactions
  7. Throughput of in-memory scoring
Evaluation rows: out-of-time chunk 489000-489999, hash buckets 9000-9199 (never used for fitting).

    .venv\\Scripts\\python.exe scripts\\evaluate_zenodo_anomaly.py
Writes data/validation/zenodo_anomaly/evaluation.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.anomaly.zenodo_detector import ZenodoAnomalyDetector  # noqa: E402
from src.anomaly.zenodo_features import MODEL_FEATURES, compute_features  # noqa: E402

FEAT = os.path.join(ROOT, "data/processed/zenodo_anomaly/features")
OUT = os.path.join(ROOT, "data/validation/zenodo_anomaly")
FIT_CHUNKS = ["481000-481999", "482000-482999", "483000-483999", "484000-484999"]
RAW_COLS = ["block_number", "tx_index", "tx_hash", "num_inputs", "num_outputs", "total_input_satoshi",
            "total_output_satoshi", "fee_satoshi", "fee_rate", "is_coinbase", "avg_input_val", "avg_output_val",
            "input_output_count_ratio", "input_output_val_ratio", "max_input_prior_tx_count",
            "avg_input_prior_tx_count", "max_input_prior_volume_satoshi", "new_address_fraction"]


def load(chunks, lo, hi, cols):
    dset = ds.dataset(FEAT, format="parquet", partitioning="hive")
    flt = (ds.field("block_chunk").isin(chunks) & (ds.field("dq_status") == "ok") & (ds.field("is_coinbase") == 0)
           & (ds.field("sample_bucket") >= lo) & (ds.field("sample_bucket") < hi))
    return dset.to_table(columns=cols, filter=flt).to_pandas()


def psi(expected_counts, actual_counts, eps=1e-6):
    e = np.asarray(expected_counts, float)
    a = np.asarray(actual_counts, float)
    e = e / e.sum() + eps
    a = a / a.sum() + eps
    return float(np.sum((a - e) * np.log(a / e)))


def compare(base_scores, other_scores, q=0.999):
    rho = spearmanr(base_scores, other_scores).statistic
    tb = base_scores >= np.quantile(base_scores, q)
    to = other_scores >= np.quantile(other_scores, q)
    jac = (tb & to).sum() / max((tb | to).sum(), 1)
    return {"spearman": round(float(rho), 4), "top_0.1pct_jaccard": round(float(jac), 4)}


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"labels_available": False,
              "note": "No reliable transaction-level illicit labels exist for this data. Nothing below is a "
                      "detection-accuracy claim; section 'synthetic_fault_injection' uses artificial anomalies."}
    prod = ZenodoAnomalyDetector.load(os.path.join(ROOT, "artifacts/models/zenodo/1.0.0"))
    summary = json.load(open(os.path.join(ROOT, "data/processed/zenodo_anomaly/score_summary.json")))

    # 1. stability across chunks -----------------------------------------------------
    fit_hist = np.sum([summary["chunks"][c]["histogram"] for c in FIT_CHUNKS], axis=0)
    report["chunk_stability"] = {
        c: {"psi_vs_fit_window": round(psi(fit_hist, v["histogram"]), 5), "flag_rate": round(v["flag_rate"], 6),
            "median_score": round(v["score_quantiles"]["p50"], 4), "p99_score": round(v["score_quantiles"]["p99"], 4),
            "role": "fit" if c in FIT_CHUNKS else ("cold_start_excluded" if c.startswith("480") else "out_of_time")}
        for c, v in summary["chunks"].items()}
    report["design_flag_rate"] = 1 - prod.flag_quantile

    # evaluation sample: out-of-time chunk, unseen buckets
    ev = load(["489000-489999"], 9000, 9200, RAW_COLS + MODEL_FEATURES)
    Xev = ev[MODEL_FEATURES].to_numpy(np.float64)
    base = prod.score(Xev)["score"]
    report["evaluation_sample"] = {"chunk": "489000-489999", "buckets": [9000, 9200], "rows": int(len(ev))}

    # 2. seed stability ---------------------------------------------------------------
    Xfit = load(FIT_CHUNKS, 0, 200, MODEL_FEATURES)[MODEL_FEATURES].to_numpy(np.float64)
    report["seed_stability"] = {}
    for seed in (1, 2, 3, 4):
        d = ZenodoAnomalyDetector(random_state=seed).fit(Xfit)
        report["seed_stability"][seed] = compare(base, d.score(Xev)["score"])

    # 3. sampling impact --------------------------------------------------------------
    report["sampling_impact"] = {}
    for buckets in (50, 500):
        Xs = load(FIT_CHUNKS, 0, buckets, MODEL_FEATURES)[MODEL_FEATURES].to_numpy(np.float64)
        t0 = time.time()
        d = ZenodoAnomalyDetector(random_state=42).fit(Xs)
        report["sampling_impact"][f"{buckets / 100:.1f}%"] = {
            "fit_rows": int(len(Xs)), "fit_seconds": round(time.time() - t0, 2), **compare(base, d.score(Xev)["score"]),
            "flag_rate_on_eval": round(float(d.score(Xev)["flagged"].mean()), 6)}
    report["sampling_impact"]["2.0% (production)"] = {"fit_rows": int(len(Xfit)),
                                                     "flag_rate_on_eval": round(float(prod.score(Xev)["flagged"].mean()), 6)}

    # 4. feature sensitivity ----------------------------------------------------------
    groups = {"address_history": ["log_max_input_prior_tx", "log_avg_input_prior_tx", "log_max_input_prior_volume",
                                  "new_addr_fraction"],
              "fee": ["log_fee", "log_fee_per_io", "log_fee_bps_of_input"],
              "shape": ["log_num_inputs", "log_num_outputs", "log_io_count_ratio"],
              "value": ["log_total_output", "log_avg_input_val", "log_avg_output_val"]}
    report["feature_sensitivity"] = {}
    for g, cols in groups.items():
        keep = [i for i, f in enumerate(MODEL_FEATURES) if f not in cols]
        d = ZenodoAnomalyDetector(random_state=42, features=[MODEL_FEATURES[i] for i in keep]).fit(Xfit[:, keep])
        report["feature_sensitivity"][f"without_{g}"] = compare(base, d.score(Xev[:, keep])["score"])

    # 5. SYNTHETIC fault injection ----------------------------------------------------
    rng = np.random.default_rng(7)
    clean = ev.sample(min(5000, len(ev)), random_state=7).reset_index(drop=True)

    def recompute(df):
        df = df.copy()
        df["total_output_satoshi"] = df["total_input_satoshi"] - df["fee_satoshi"]
        df["fee_rate"] = df["fee_satoshi"] / (df["num_inputs"] + df["num_outputs"])
        df["avg_input_val"] = df["total_input_satoshi"] / df["num_inputs"]
        df["avg_output_val"] = df["total_output_satoshi"] / df["num_outputs"]
        df["input_output_count_ratio"] = df["num_inputs"] / df["num_outputs"]
        df["input_output_val_ratio"] = df["total_input_satoshi"] / df["total_output_satoshi"].clip(lower=1)
        return df

    def fee_spike(df):  # absurd fee: 30% of the input value
        df = df.copy()
        df["fee_satoshi"] = (df["total_input_satoshi"] * 0.30).astype("int64")
        return recompute(df)

    def fan_in(df):  # consolidation of 150-400 inputs into one output, all from fresh addresses
        df = df.copy()
        df["num_inputs"] = rng.integers(150, 400, len(df))
        df["num_outputs"] = 1
        df["new_address_fraction"] = 1.0
        df["max_input_prior_tx_count"] = 0
        df["avg_input_prior_tx_count"] = 0.0
        df["max_input_prior_volume_satoshi"] = 0
        return recompute(df)

    def fan_out(df):  # peel/spray: one input to 200-1000 outputs
        df = df.copy()
        df["num_inputs"] = 1
        df["num_outputs"] = rng.integers(200, 1000, len(df))
        return recompute(df)

    def whale(df):  # value x10,000 with typical fee
        df = df.copy()
        df["total_input_satoshi"] = df["total_input_satoshi"] * 10_000
        return recompute(df)

    def mild(df):  # deliberately subtle: fee x2
        df = df.copy()
        df["fee_satoshi"] = df["fee_satoshi"] * 2
        return recompute(df)

    cs = prod.score(compute_features(clean).to_numpy())["score"]
    report["synthetic_fault_injection"] = {"clean_rows": int(len(clean)),
                                           "clean_flag_rate": round(float((cs >= prod.threshold).mean()), 5)}
    for name, fn in [("fee_spike_30pct", fee_spike), ("fan_in_150_400_fresh_inputs", fan_in),
                     ("fan_out_200_1000_outputs", fan_out), ("value_x10000", whale), ("fee_x2_subtle", mild)]:
        inj = fn(clean)
        s = prod.score(compute_features(inj).to_numpy())["score"]
        y = np.r_[np.zeros(len(cs)), np.ones(len(s))]
        report["synthetic_fault_injection"][name] = {
            "flag_rate_at_threshold": round(float((s >= prod.threshold).mean()), 4),
            "auroc_injected_vs_clean": round(float(roc_auc_score(y, np.r_[cs, s])), 4),
            "median_score": round(float(np.median(s)), 4)}

    # 6. top-ranked inspection --------------------------------------------------------
    top = pd.read_parquet(os.path.join(ROOT, "data/processed/zenodo_anomaly/top_ranked.parquet"))
    raw = ds.dataset(os.path.join(ROOT, "data/parquet/zenodo_features"), format="parquet", partitioning="hive")
    top_raw = raw.to_table(columns=RAW_COLS, filter=ds.field("tx_hash").isin(top["tx_hash"].head(200).tolist())).to_pandas()
    reason_first = top["reason_codes"].str.split(";").str[0].str.split(":").str[0].value_counts().head(8).to_dict()
    report["top_ranked_inspection"] = {
        "n_inspected": 200,
        "block_range": [int(top_raw.block_number.min()), int(top_raw.block_number.max())],
        "share_in_blocks_488000_488099": round(float(top_raw.block_number.between(488000, 488099).mean()), 3),
        "num_inputs_median": float(top_raw.num_inputs.median()), "num_outputs_median": float(top_raw.num_outputs.median()),
        "new_address_fraction_median": float(top_raw.new_address_fraction.median()),
        "fee_satoshi_median": float(top_raw.fee_satoshi.median()),
        "top1000_primary_reason_counts": reason_first,
        "top1000_by_chunk": (top.block_number // 1000 * 1000).value_counts().sort_index().astype(int)
                            .rename(lambda b: f"{b}-{b + 999}").to_dict(),
        "examples": top_raw.sort_values("num_inputs", ascending=False).head(5)[
            ["block_number", "tx_index", "tx_hash", "num_inputs", "num_outputs", "total_input_satoshi", "fee_satoshi",
             "new_address_fraction"]].to_dict("records"),
    }

    # 7. throughput ------------------------------------------------------------------
    t0 = time.perf_counter()
    for _ in range(3):
        prod.score(Xev)
    dt = (time.perf_counter() - t0) / 3
    report["throughput"] = {"in_memory_rows_per_second": round(len(Xev) / dt, 1), "rows": int(len(Xev)),
                            "batch_pipeline_rows_per_second_incl_io": summary["rows_per_second"],
                            "batch_pipeline_total_seconds": summary["score_seconds"]}
    json.dump(report, open(os.path.join(OUT, "evaluation.json"), "w"), indent=2, default=str)
    print(json.dumps(report, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
