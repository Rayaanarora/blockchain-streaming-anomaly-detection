"""
Preprocessing metrics for notebook 01.

Main part - the Elliptic Spark pipeline that feeds TGAT-BoostNet (src/spark/elliptic_spark.py):
  * feature-level agreement with an independent pandas reference (src/evaluation/data.py),
  * integrity counts, data quality before/after, class balance,
  * causality / leakage checks, runtime and memory, step-subset scaling.

Appendix - the Zenodo Spark pipeline (src/spark/zenodo_batch.py) against the repository's pure-Python
causal reference (scripts/run_spark_scale_benchmark.py::compute_ground_truth_for_keys), on a raw slice.
"""
import bz2
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.data import REPO_ROOT

ZENODO_FEATURES = ["num_inputs", "num_outputs", "total_input_satoshi", "total_output_satoshi", "fee_satoshi",
                   "fee_rate", "is_coinbase", "avg_input_val", "avg_output_val", "input_output_count_ratio",
                   "input_output_val_ratio", "max_input_prior_tx_count", "avg_input_prior_tx_count",
                   "max_input_prior_volume_satoshi", "new_address_fraction"]
ZENODO_PART_URL = "https://zenodo.org/records/7157854/files/btc-tx-480000-489999-aa.tar.bz2?download=1"
ZENODO_LOCAL_RAW = os.path.join(REPO_ROOT, "data", "raw", "sample_a", "btc-tx-480000-489999-aa")


# --------------------------------------------------------------------------- agreement metrics
def agreement_table(ref: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Per-feature error of ``test`` against ``ref`` (rows already aligned on the same keys).

    The course's preprocessing metrics, with e = test - ref:
      MSE = mean(e^2); RMSE = sqrt(MSE); MAE = mean|e|; MedAE = median|e|; ME (bias) = mean(e);
      MAPE = 100 * mean(|e| / |ref|) over rows with ref != 0 (zero references have no percentage error);
      NRMSE = RMSE / (max(ref) - min(ref)) (range-normalised; 0 when the feature is constant and e = 0);
      R^2 = 1 - SSE / SST.
    Plus max |error|, relative error = max|e| / max|ref|, exact-match rate (bitwise equal values) and
    Lin's concordance correlation coefficient (CCC).
    """
    rows = {}
    for c in cols:
        a = ref[c].to_numpy(np.float64)
        b = test[c].to_numpy(np.float64)
        e = b - a
        ss_tot = ((a - a.mean()) ** 2).sum()
        va, vb = a.var(), b.var()
        ccc = (2 * np.cov(a, b, bias=True)[0, 1] / (va + vb + (a.mean() - b.mean()) ** 2)) if va + vb > 0 else 1.0
        nz = a != 0
        rmse = np.sqrt((e ** 2).mean())
        rng = a.max() - a.min()
        rows[c] = {"MSE": (e ** 2).mean(), "RMSE": rmse, "MAE": np.abs(e).mean(), "MedAE": float(np.median(np.abs(e))),
                   "ME_bias": e.mean(), "MAPE_pct": 100 * float(np.mean(np.abs(e[nz]) / np.abs(a[nz]))) if nz.any() else 0.0,
                   "NRMSE": rmse / rng if rng > 0 else (0.0 if rmse == 0 else np.inf),
                   "max_abs_err": np.abs(e).max(),
                   "rel_err": np.abs(e).max() / max(np.abs(a).max(), 1e-300),
                   "R2": 1 - (e ** 2).sum() / ss_tot if ss_tot > 0 else (1.0 if np.all(e == 0) else 0.0),
                   "exact_match_rate": float((a == b).mean()), "CCC": ccc,
                   "null_rate": float(test[c].isna().mean())}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- data quality
def quality_profile(X: np.ndarray, names: List[str]) -> pd.DataFrame:
    """Missingness, outlier fractions (|z| > 3; outside 1.5 IQR fences), skewness, excess kurtosis."""
    X = np.asarray(X, dtype=np.float64)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[sd == 0] = 1
    z = np.abs((X - mu) / sd)
    q1, q3 = np.nanpercentile(X, 25, 0), np.nanpercentile(X, 75, 0)
    iqr = q3 - q1
    out_iqr = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).mean(0)
    return pd.DataFrame({"missing_rate": np.isnan(X).mean(0), "outlier_z3": (z > 3).mean(0),
                         "outlier_iqr": out_iqr, "skewness": stats.skew(X, 0, nan_policy="omit"),
                         "excess_kurtosis": stats.kurtosis(X, 0, nan_policy="omit")}, index=names)


def class_snr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Class signal-to-noise ratio per feature: |mu_illicit - mu_licit| / sqrt((var_illicit + var_licit) / 2).

    A standardised mean difference (Cohen's d with pooled variance). It is invariant to any affine
    transform, so standardisation leaves it unchanged; a non-linear transform (asinh) can change it.
    """
    a, b = X[y == 1].astype(np.float64), X[y == 0].astype(np.float64)
    pooled = np.sqrt((a.var(0) + b.var(0)) / 2)
    pooled[pooled == 0] = np.inf
    return np.abs(a.mean(0) - b.mean(0)) / pooled


def coefficient_of_variation_snr(X: np.ndarray) -> np.ndarray:
    """Classical SNR = |mean| / std per feature (reported for completeness; ill-defined for centred data)."""
    sd = X.std(0)
    sd[sd == 0] = np.inf
    return np.abs(X.mean(0)) / sd


def signed_log(X: np.ndarray) -> np.ndarray:
    """asinh transform: sign-preserving log-like compression of heavy tails."""
    return np.arcsinh(X)


# --------------------------------------------------------------------------- Elliptic Spark
def read_spark_elliptic(spark_dir: str) -> Dict[str, pd.DataFrame]:
    nodes = pd.read_parquet(os.path.join(spark_dir, "nodes")).sort_values("node_idx").reset_index(drop=True)
    scaled = pd.read_parquet(os.path.join(spark_dir, "nodes_scaled")).sort_values("node_idx").reset_index(drop=True)
    edges = pd.read_parquet(os.path.join(spark_dir, "edges"))
    return {"nodes": nodes, "scaled": scaled, "edges": edges}


def reference_scaled(X_raw: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Independent float64 reference of the Spark scaler: (x - mean) / pop-std, stats from fit_mask rows."""
    mu = X_raw[fit_mask].astype(np.float64).mean(0)
    sd = X_raw[fit_mask].astype(np.float64).std(0)
    sd[sd == 0] = 1.0
    return (X_raw.astype(np.float64) - mu) / sd


def raw_csv_integrity(raw_dir: str) -> Dict[str, float]:
    """Line-level counts on the raw CSVs, independent of both pipelines."""
    out = {}
    f = os.path.join(raw_dir, "elliptic_txs_features.csv")
    n, bad, nonnum = 0, 0, 0
    with open(f, "r") as fh:
        for line in fh:
            n += 1
            parts = line.rstrip("\n").split(",")
            if len(parts) != 167:
                bad += 1
                continue
            try:
                [float(x) for x in parts]
            except ValueError:
                nonnum += 1
    out.update(feature_lines=n, malformed_lines=bad, non_numeric_lines=nonnum)
    cls = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_classes.csv"))
    edges = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_edgelist.csv"))
    feats_ids = pd.read_csv(f, header=None, usecols=[0, 1])
    ids = set(feats_ids[0])
    out.update(class_rows=len(cls), invalid_labels=int((~cls.iloc[:, 1].astype(str).isin(["1", "2", "unknown"])).sum()),
               edge_rows=len(edges),
               dangling_edges=int((~edges.iloc[:, 0].isin(ids) | ~edges.iloc[:, 1].isin(ids)).sum()),
               duplicate_edges=int(edges.duplicated().sum()), self_loops=int((edges.iloc[:, 0] == edges.iloc[:, 1]).sum()),
               duplicate_tx_ids=int(feats_ids[0].duplicated().sum()),
               time_step_min=int(feats_ids[1].min()), time_step_max=int(feats_ids[1].max()))
    return out


def memory_tracker():
    """The repository's 50 ms psutil tracker (driver / JVM children / process tree)."""
    from scripts.run_spark_scale_benchmark import ContinuousMemoryTracker
    return ContinuousMemoryTracker(sample_interval_sec=0.05)


def dir_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def benchmark_elliptic_spark(raw_dir: str, max_step: int = 49, master: str = "local[4]",
                             work_dir: Optional[str] = None) -> Dict[str, float]:
    """Run the Elliptic Spark pipeline on time steps 1..max_step and measure time, memory and disk.

    For max_step < 49 the three CSVs are first filtered to those steps (edges are intra-step).
    """
    from src.spark.elliptic_spark import run
    from src.spark.session import get_spark
    scratch = os.path.join(REPO_ROOT, "data", "temp")
    os.makedirs(scratch, exist_ok=True)
    work = work_dir or tempfile.mkdtemp(prefix="elliptic_bench_", dir=scratch)
    os.makedirs(work, exist_ok=True)
    src_dir = raw_dir
    if max_step < 49:
        src_dir = os.path.join(work, f"raw_steps_1_{max_step}")
        os.makedirs(src_dir, exist_ok=True)
        f = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_features.csv"), header=None)
        f = f[f[1] <= max_step]
        f.to_csv(os.path.join(src_dir, "elliptic_txs_features.csv"), header=False, index=False)
        ids = set(f[0])
        c = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_classes.csv"))
        c[c.iloc[:, 0].isin(ids)].to_csv(os.path.join(src_dir, "elliptic_txs_classes.csv"), index=False)
        e = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_edgelist.csv"))
        e[e.iloc[:, 0].isin(ids)].to_csv(os.path.join(src_dir, "elliptic_txs_edgelist.csv"), index=False)
    in_bytes = sum(os.path.getsize(os.path.join(src_dir, n)) for n in
                   ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"))
    out_dir = os.path.join(work, f"out_steps_1_{max_step}")
    mem = memory_tracker()
    mem.start()
    t0 = time.perf_counter()
    spark = get_spark(f"EllipticBench{max_step}", master=master)
    t_session = time.perf_counter() - t0
    try:
        res = run(spark, src_dir, out_dir, overwrite=True, expected_time_steps=None if max_step < 49 else (1, 49, 49))
    finally:
        spark.stop()
    elapsed = time.perf_counter() - t0
    mem.stop()
    n_nodes, n_edges = res["manifest"]["n_nodes"], res["manifest"]["n_edges"]
    n_lines = n_nodes + n_nodes + n_edges  # feature rows + class rows + edge rows
    out = {"max_step": max_step, "n_nodes": n_nodes, "n_edges": n_edges, "input_lines": n_lines,
           "input_mb": in_bytes / 2 ** 20, "output_mb": dir_bytes(out_dir) / 2 ** 20,
           "wall_seconds": elapsed, "spark_session_seconds": t_session, "pipeline_seconds": res["manifest"]["elapsed_seconds"],
           "lines_per_s": n_lines / elapsed, "tx_per_s": n_nodes / elapsed,
           "driver_peak_rss_mb": mem.driver_overall_peak_rss_mb, "jvm_peak_rss_mb": mem.jvm_child_peak_rss_mb,
           "tree_peak_rss_mb": mem.process_tree_peak_rss_mb}
    shutil.rmtree(work, ignore_errors=True)
    return out


def benchmark_pandas_reference(raw_dir: str) -> Dict[str, float]:
    """Same measurements for the pandas reference loader (single process)."""
    import psutil
    from src.evaluation.data import load_elliptic, standardise
    proc = psutil.Process()
    r0 = proc.memory_info().rss
    t0 = time.perf_counter()
    d = load_elliptic(raw_dir, cache=False, verbose=False)
    standardise(d)
    el = time.perf_counter() - t0
    return {"wall_seconds": el, "tx_per_s": d.n / el, "rss_growth_mb": (proc.memory_info().rss - r0) / 2 ** 20}


# --------------------------------------------------------------------------- Zenodo appendix
def zenodo_slice(n_lines: int, out_path: str) -> str:
    """Write the first ``n_lines`` raw lines of part 'aa' to ``out_path``.

    Uses the local raw file when present; otherwise streams the Zenodo tar.bz2 and stops after
    ``n_lines`` (only the first few MB are downloaded).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(ZENODO_LOCAL_RAW):
        with open(ZENODO_LOCAL_RAW, "r", encoding="ascii") as fi, open(out_path, "w", encoding="ascii") as fo:
            for _ in range(n_lines):
                ln = fi.readline()
                if not ln:
                    break
                fo.write(ln)
        return out_path
    with urllib.request.urlopen(ZENODO_PART_URL) as resp:
        tf = tarfile.open(fileobj=resp, mode="r|bz2")
        for member in tf:
            fh = tf.extractfile(member)
            if fh is None:
                continue
            with open(out_path, "w", encoding="ascii") as fo:
                for i, ln in enumerate(fh):
                    if i >= n_lines:
                        break
                    fo.write(ln.decode("ascii"))
            break
    return out_path


def zenodo_reference(slice_path: str) -> pd.DataFrame:
    """All completed transactions of the slice, computed by the pure-Python causal reference."""
    from scripts.run_spark_scale_benchmark import compute_ground_truth_for_keys, parse_zenodo_line
    keys = set()
    with open(slice_path, "r", encoding="ascii") as f:
        for ln in f:
            r = parse_zenodo_line(ln)
            if r:
                keys.add((r[1], r[2]))
    gt = compute_ground_truth_for_keys(slice_path, keys)
    df = pd.DataFrame(list(gt.values()))
    return df.sort_values(["block_number", "tx_index"]).reset_index(drop=True)


def zenodo_spark(slice_path: str, master: str = "local[2]", tmp: Optional[str] = None):
    """Spark features of the slice (collected; slices are small) plus line/record counts."""
    from src.spark.zenodo_batch import ZenodoSparkBatchPipeline, build_spark_session
    import pyspark.sql.functions as F
    from src.spark.session import resolve_java_home
    os.environ["JAVA_HOME"] = resolve_java_home()  # the shell's JAVA_HOME may point at a removed JDK
    tmp = tmp or tempfile.mkdtemp(prefix="zen_spark_")
    t0 = time.perf_counter()
    spark = build_spark_session("ZenodoAppendix", master, os.path.join(tmp, "tmp"), os.path.join(tmp, "wh"))
    try:
        pipe = ZenodoSparkBatchPipeline(spark)
        raw = spark.read.text([slice_path])
        n_raw = raw.count()
        rec = pipe.parse_raw_zenodo_text(raw)
        n_rec = rec.count()
        n_groups = rec.select("block_number", "tx_index").distinct().count()
        tx = pipe.assemble_completed_transactions(rec)
        feats = pipe.compute_leakage_safe_features(rec, tx)
        pdf = feats.drop("block_chunk").toPandas()
        elapsed = time.perf_counter() - t0
    finally:
        spark.stop()
    counts = {"raw_lines": n_raw, "valid_records": n_rec, "malformed_lines": n_raw - n_rec,
              "tx_groups": n_groups, "completed_tx": len(pdf), "dropped_incomplete_tx": n_groups - len(pdf),
              "spark_seconds": elapsed}
    pdf = pdf.sort_values(["block_number", "tx_index"]).reset_index(drop=True)
    return pdf, counts


def zenodo_sanity(df: pd.DataFrame) -> Dict[str, int]:
    nc = df["is_coinbase"] == 0
    return {"negative_fee": int((df.loc[nc, "fee_satoshi"] < 0).sum()),
            "output_exceeds_input": int((df.loc[nc, "total_output_satoshi"] > df.loc[nc, "total_input_satoshi"]).sum()),
            "negative_fee_rate": int((df["fee_rate"] < 0).sum()),
            "zero_output_value_tx": int((df["total_output_satoshi"] == 0).sum()),
            "new_address_fraction_out_of_[0,1]": int(((df["new_address_fraction"] < 0) |
                                                       (df["new_address_fraction"] > 1)).sum())}


def zenodo_causality_test(slice_path: str, cut_fraction: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Perturb every raw line after a cut point (values scrambled, addresses replaced) and recompute.

    A causal pipeline must give bit-identical features for all transactions that end before the cut.
    Returns per-feature max |change| for pre-cut transactions under (a) the causal reference and
    (b) a deliberately non-causal variant that counts an address's *total* history (past + future).
    """
    from scripts.run_spark_scale_benchmark import parse_zenodo_line
    with open(slice_path, "r", encoding="ascii") as f:
        lines = f.readlines()
    cut = int(len(lines) * cut_fraction)
    cut_key = parse_zenodo_line(lines[cut])[1:3]
    rng = np.random.default_rng(seed)
    pert = list(lines[:cut])
    for ln in lines[cut:]:
        k, b, t, s, d, v = ln.split()
        if k == "BTC-IN":
            s = s if rng.random() < 0.5 else f"1Fake{rng.integers(1e9)}"
        else:
            d = d if rng.random() < 0.5 else f"1Fake{rng.integers(1e9)}"
        pert.append(f"{k} {b} {t} {s} {d} {int(int(v) * rng.uniform(0.1, 10))}\n")
    tmpd = tempfile.mkdtemp(prefix="zen_causal_")
    p_path = os.path.join(tmpd, "perturbed.txt")
    with open(p_path, "w", encoding="ascii") as f:
        f.writelines(pert)
    a, b = zenodo_reference(slice_path), zenodo_reference(p_path)
    key = ["block_number", "tx_index"]
    # transactions fully before the cut (their key sorts before the first perturbed record)
    def before(df):
        return df[(df.block_number < cut_key[0]) | ((df.block_number == cut_key[0]) & (df.tx_index < cut_key[1]))]
    ma = before(a).merge(before(b), on=key, suffixes=("", "_p"))
    causal = {c: float(np.abs(ma[c].astype(float) - ma[c + "_p"].astype(float)).max()) for c in ZENODO_FEATURES}
    na, nb = _noncausal_prior_count(lines), _noncausal_prior_count(pert)
    m2 = before(a)[key].merge(na, on=key).merge(nb, on=key, suffixes=("", "_p"))
    noncausal = float(np.abs(m2["total_count"] - m2["total_count_p"]).max())
    shutil.rmtree(tmpd, ignore_errors=True)
    out = pd.DataFrame({"causal_reference_max_abs_change": causal})
    out.loc["max_input_total_tx_count (non-causal control)", "causal_reference_max_abs_change"] = np.nan
    out["noncausal_control_max_abs_change"] = np.nan
    out.loc["max_input_total_tx_count (non-causal control)", "noncausal_control_max_abs_change"] = noncausal
    out.attrs.update(cut_line=cut, cut_key=cut_key, n_pre_cut_tx=len(ma))
    return out


def _noncausal_prior_count(lines) -> pd.DataFrame:
    """Leaky control: for each tx, max over its input addresses of the address's total tx count (any time)."""
    from collections import defaultdict
    from scripts.run_spark_scale_benchmark import parse_zenodo_line
    addr_tx, tx_inputs = defaultdict(set), defaultdict(set)
    for ln in lines:
        r = parse_zenodo_line(ln)
        if not r:
            continue
        kind, b, t, _, addr, _ = r
        addr_tx[addr].add((b, t))
        if kind == "BTC-IN":
            tx_inputs[(b, t)].add(addr)
    rows = [(b, t, max(len(addr_tx[a]) for a in ins)) for (b, t), ins in tx_inputs.items()]
    return pd.DataFrame(rows, columns=["block_number", "tx_index", "total_count"])


def load_scale_reports() -> pd.DataFrame:
    """The existing Zenodo Spark reports (50k / 100k bounded slices, v3; full 85.7 M-line run). Loaded, not retyped."""
    import json
    v = os.path.join(REPO_ROOT, "data", "validation")
    rows = []
    for n in (50000, 100000):
        with open(os.path.join(v, f"spark_scale_benchmark_{n}_v3.json")) as f:
            d = json.load(f)
        rows.append({"run": f"{n // 1000}k-line slice", "raw_lines": d["raw_lines_processed"],
                     "transactions": d["completed_transactions"], "wall_seconds": d["wall_clock_elapsed_seconds"],
                     "lines_per_s": d["throughput_lines_per_sec"], "tx_per_s": d["throughput_transactions_per_sec"],
                     "driver_peak_rss_mb": d["memory_metrics"]["driver_overall_peak_rss_mb"],
                     "jvm_peak_rss_mb": d["memory_metrics"]["jvm_child_peak_rss_mb"],
                     "tree_peak_rss_mb": d["memory_metrics"]["process_tree_peak_rss_mb"],
                     "output_mb": d["parquet_output_size_mb"], "null_values": sum(d["null_counts"].values()),
                     "source": f"spark_scale_benchmark_{n}_v3.json"})
    with open(os.path.join(v, "zenodo_preprocessing_report.json")) as f:
        d = json.load(f)
    rows.append({"run": "full (18 parts)", "raw_lines": d["input_statistics"]["raw_lines_read"],
                 "transactions": d["input_statistics"]["completed_transactions"],
                 "wall_seconds": d["performance_metrics"]["total_elapsed_seconds"],
                 "lines_per_s": d["performance_metrics"]["lines_per_second"],
                 "tx_per_s": d["input_statistics"]["completed_transactions"] / d["performance_metrics"]["total_elapsed_seconds"],
                 "driver_peak_rss_mb": d["memory_profile_mb"]["driver_overall_peak_rss_mb"],
                 "jvm_peak_rss_mb": d["memory_profile_mb"]["jvm_child_peak_rss_mb"],
                 "tree_peak_rss_mb": d["memory_profile_mb"]["process_tree_peak_rss_mb"],
                 "output_mb": d["output_statistics"]["total_output_mb"],
                 "null_values": d["validation_summary"]["null_key_violations"],
                 "source": "zenodo_preprocessing_report.json"})
    return pd.DataFrame(rows).set_index("run")
