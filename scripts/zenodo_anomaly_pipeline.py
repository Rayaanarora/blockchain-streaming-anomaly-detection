"""
Zenodo unsupervised anomaly-scoring pipeline (reuses data/parquet/zenodo_features; the raw
reconstruction is NOT rerun).

Stages (run separately or with ``all``):
  features  Spark: validate schema + row-level data quality, derive model features and a
            deterministic sample bucket -> data/processed/zenodo_anomaly/features/
  fit       Fit the detector on a hash sample of the fit window -> artifacts/models/zenodo/<version>/
  score     Score every transaction (PyArrow batches, bounded memory) ->
            data/processed/zenodo_anomaly/scores/, top_ranked.parquet, score_summary.json

    .venv\\Scripts\\python.exe scripts\\zenodo_anomaly_pipeline.py features
    .venv\\Scripts\\python.exe scripts\\zenodo_anomaly_pipeline.py fit
    .venv\\Scripts\\python.exe scripts\\zenodo_anomaly_pipeline.py score

Fit window default: blocks 481000-484999. Chunk 480000-480999 is excluded because address
history starts at block 480000 in this dataset (cold start: almost every address looks "new").
Blocks 485000-489999 are never used for fitting and serve as an out-of-time window.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.anomaly.zenodo_detector import ZenodoAnomalyDetector  # noqa: E402
from src.anomaly.zenodo_features import (FEATURE_VERSION, INPUT_COLUMNS, MODEL_FEATURES,  # noqa: E402
                                         sample_bucket_spark, spark_dq_status, spark_feature_columns)

SRC = os.path.join(ROOT, "data/parquet/zenodo_features")
OUT = os.path.join(ROOT, "data/processed/zenodo_anomaly")
MODEL_ROOT = os.path.join(ROOT, "artifacts/models/zenodo")
FIT_CHUNKS = ["481000-481999", "482000-482999", "483000-483999", "484000-484999"]
ID_COLS = ["block_number", "tx_index", "tx_hash", "is_coinbase"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stage_features(args):
    from src.spark.session import get_spark, to_spark_path
    import pyspark.sql.functions as F

    out_dir = os.path.join(args.out, "features")
    if os.path.exists(out_dir) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists (use --overwrite; old output is moved to a backup)")
    spark = get_spark("ZenodoAnomalyFeatures", master=args.master, shuffle_partitions=16)
    try:
        t0 = time.time()
        df = spark.read.parquet(to_spark_path(args.src))
        expected = set(INPUT_COLUMNS) | {"block_chunk"}
        missing, extra = expected - set(df.columns), set(df.columns) - expected
        if missing:
            raise RuntimeError(f"Zenodo Parquet schema missing columns: {sorted(missing)}")
        feat = df.select(*INPUT_COLUMNS, "block_chunk", spark_dq_status(F).alias("dq_status"),
                         sample_bucket_spark(F).alias("sample_bucket"), *spark_feature_columns(F))
        staging = out_dir + f".staging_{int(time.time())}"
        feat.write.mode("overwrite").partitionBy("block_chunk").parquet(to_spark_path(staging))
        written = spark.read.parquet(to_spark_path(staging))
        n_src, n_out = df.count(), written.count()
        dq = {f"{r['block_chunk']}|{r['dq_status']}": r["n"] for r in
              written.groupBy("block_chunk", "dq_status").agg(F.count("*").alias("n")).collect()}
        n_distinct_hash = written.select("tx_hash").distinct().count()
        nonfinite = written.select([F.sum(F.when(F.isnan(c) | F.col(c).isNull() |
                                                 (F.abs(F.col(c)) == float("inf")), 1).otherwise(0)).alias(c)
                                    for c in MODEL_FEATURES]).first().asDict()
        coinbase = written.filter(F.col("is_coinbase") == 1).count()
        elapsed = time.time() - t0
    finally:
        spark.stop()
    if n_src != n_out:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"row count mismatch after feature write: {n_src} vs {n_out}")
    if os.path.exists(out_dir):
        shutil.move(out_dir, out_dir + f".backup_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.move(staging, out_dir)
    report = {"feature_version": FEATURE_VERSION, "source": args.src, "rows_in": n_src, "rows_out": n_out,
              "extra_source_columns": sorted(extra), "distinct_tx_hash": n_distinct_hash,
              "duplicate_tx_hash_rows": n_out - n_distinct_hash, "coinbase_rows": coinbase,
              "non_finite_feature_values": nonfinite, "dq_counts_by_chunk_status": dq,
              "spark_seconds": round(elapsed, 1)}
    json.dump(report, open(os.path.join(args.out, "features_report.json"), "w"), indent=2)
    log(json.dumps({k: v for k, v in report.items() if k != "dq_counts_by_chunk_status"}, indent=1))


def load_fit_matrix(features_dir, chunks, bucket_lo, bucket_hi):
    dset = ds.dataset(features_dir, format="parquet", partitioning="hive")
    flt = (ds.field("block_chunk").isin(chunks) & (ds.field("dq_status") == "ok") & (ds.field("is_coinbase") == 0)
           & (ds.field("sample_bucket") >= bucket_lo) & (ds.field("sample_bucket") < bucket_hi))
    t = dset.to_table(columns=MODEL_FEATURES, filter=flt)
    return np.column_stack([t[c].to_numpy() for c in MODEL_FEATURES]).astype(np.float64)


def stage_fit(args):
    fdir = os.path.join(args.out, "features")
    X = load_fit_matrix(fdir, args.fit_chunks, 0, args.sample_buckets)
    log(f"fit sample: {len(X):,} rows (buckets 0..{args.sample_buckets - 1} of 10000, chunks {args.fit_chunks})")
    det = ZenodoAnomalyDetector(n_estimators=args.n_estimators, max_samples=args.max_samples,
                                flag_quantile=args.flag_quantile, random_state=args.seed)
    det.fit(X, meta={"fit_chunks": args.fit_chunks, "sample_buckets": [0, args.sample_buckets],
                     "sampling": "tx_hash[0:8] as hex mod 10000 < sample_buckets; dq_status=ok; non-coinbase"})
    path = os.path.join(args.model_root, args.model_version)
    if os.path.exists(os.path.join(path, "manifest.json")) and not args.overwrite:
        raise FileExistsError(f"{path} exists; bump --model-version")
    det.save(path, args.model_version)
    log(f"saved {path}; threshold={det.threshold:.5f}; fit {det.meta['fit_seconds']}s")


def stage_score(args):
    det = ZenodoAnomalyDetector.load(os.path.join(args.model_root, args.model_version))
    fdir = os.path.join(args.out, "features")
    sdir = os.path.join(args.out, "scores")
    if os.path.exists(sdir) and not args.overwrite:
        raise FileExistsError(f"{sdir} exists (use --overwrite)")
    staging = sdir + f".staging_{int(time.time())}"
    os.makedirs(staging)
    dset = ds.dataset(fdir, format="parquet", partitioning="hive")
    chunks = sorted({os.path.basename(os.path.dirname(f)).split("=", 1)[1] for f in dset.files})
    schema = pa.schema([("block_number", pa.int64()), ("tx_index", pa.int32()), ("tx_hash", pa.string()),
                        ("is_coinbase", pa.int32()), ("dq_status", pa.string()), ("score_status", pa.string()),
                        ("score", pa.float32()), ("percentile", pa.float32()), ("flagged", pa.bool_()),
                        ("reason_codes", pa.string())])
    top_frames, summary, t_all = [], {"chunks": {}}, time.time()
    total = 0
    for chunk in chunks:
        t0 = time.time()
        os.makedirs(os.path.join(staging, f"block_chunk={chunk}"))
        writer = pq.ParquetWriter(os.path.join(staging, f"block_chunk={chunk}", "part-0.parquet"), schema)
        scores_chunk, n_rows, n_flag, status_counts = [], 0, 0, {}
        scanner = dset.scanner(columns=ID_COLS + ["dq_status"] + MODEL_FEATURES,
                               filter=ds.field("block_chunk") == chunk, batch_size=200_000)
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            b = batch.to_pandas()
            n = len(b)
            score = np.full(n, np.nan, np.float32)
            pct = np.full(n, np.nan, np.float32)
            flag = np.zeros(n, bool)
            reasons = np.full(n, "", dtype=object)
            status = np.where(b["dq_status"] != "ok", "skipped_invalid_input",
                              np.where(b["is_coinbase"] == 1, "skipped_coinbase", "scored")).astype(object)
            m = status == "scored"
            if m.any():
                X = b.loc[m, MODEL_FEATURES].to_numpy(np.float64)
                r = det.score(X, reasons_min_percentile=0.99)
                score[m], pct[m], flag[m], reasons[m] = r["score"], r["percentile"], r["flagged"], r["reason_codes"]
            out = pd.DataFrame({"block_number": b["block_number"], "tx_index": b["tx_index"], "tx_hash": b["tx_hash"],
                                "is_coinbase": b["is_coinbase"], "dq_status": b["dq_status"], "score_status": status,
                                "score": score, "percentile": pct, "flagged": flag, "reason_codes": reasons})
            writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))
            n_rows += n
            n_flag += int(flag.sum())
            for k, v in pd.Series(status).value_counts().items():
                status_counts[k] = status_counts.get(k, 0) + int(v)
            sc = score[m]
            scores_chunk.append(sc)
            top_frames.append(out[m].nlargest(200, "score"))
        writer.close()
        s = np.concatenate(scores_chunk) if scores_chunk else np.array([])
        q = np.quantile(s, [0.01, 0.25, 0.5, 0.75, 0.99, 0.999]) if len(s) else []
        summary["chunks"][chunk] = {"rows": n_rows, "status": status_counts, "flagged": n_flag,
                                    "flag_rate": n_flag / max(status_counts.get("scored", 0), 1),
                                    "score_quantiles": dict(zip(["p01", "p25", "p50", "p75", "p99", "p999"],
                                                                map(float, q))),
                                    "score_mean": float(s.mean()) if len(s) else None,
                                    "seconds": round(time.time() - t0, 1)}
        # keep a compact histogram for stability analysis
        summary["chunks"][chunk]["histogram"] = np.histogram(s, bins=50, range=(0.3, 0.85))[0].tolist()
        total += n_rows
        log(f"{chunk}: {n_rows:,} rows, flagged {n_flag:,}, {summary['chunks'][chunk]['seconds']}s")
    if os.path.exists(sdir):
        shutil.move(sdir, sdir + f".backup_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.move(staging, sdir)
    elapsed = time.time() - t_all
    top = pd.concat(top_frames).nlargest(1000, "score")
    top.to_parquet(os.path.join(args.out, "top_ranked.parquet"), index=False)
    summary.update({"model_id": det.manifest["model_id"], "model_version": det.manifest["model_version"],
                    "threshold": det.threshold, "total_rows": total, "score_seconds": round(elapsed, 1),
                    "rows_per_second": round(total / elapsed, 1),
                    "histogram_bins": np.linspace(0.3, 0.85, 51).round(4).tolist()})
    json.dump(summary, open(os.path.join(args.out, "score_summary.json"), "w"), indent=2)
    log(f"scored {total:,} rows in {elapsed:.0f}s ({total / elapsed:,.0f} rows/s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["features", "fit", "score", "all"])
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model-root", default=MODEL_ROOT)
    ap.add_argument("--model-version", default="1.0.0")
    ap.add_argument("--fit-chunks", nargs="+", default=FIT_CHUNKS)
    ap.add_argument("--sample-buckets", type=int, default=200, help="fit on buckets [0, N) of 10000 (200 = 2%%)")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--flag-quantile", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--master", default="local[4]")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stages = ["features", "fit", "score"] if args.stage == "all" else [args.stage]
    for st in stages:
        log("== stage", st)
        {"features": stage_features, "fit": stage_fit, "score": stage_score}[st](args)


if __name__ == "__main__":
    main()
