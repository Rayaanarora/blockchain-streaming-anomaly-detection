"""
Apache Spark preprocessing for the Elliptic Bitcoin dataset.

Input (Kaggle layout, data/raw/elliptic/):
  elliptic_txs_features.csv  no header; txId, time_step, 165 features
  elliptic_txs_classes.csv   header txId,class; class in {"1","2","unknown"}
  elliptic_txs_edgelist.csv  header txId1,txId2 (directed: txId1 -> txId2)

Output (data/processed/elliptic_spark/):
  nodes/          Parquet, partitioned by time_step: node_idx, tx_id, label, label_status,
                  split, f1..f165 (raw float32 values)
  nodes_scaled/   Parquet, partitioned by time_step: node_idx, tx_id, label, split,
                  f1..f165 standardised with mean/std from labelled train rows only
  edges/          Parquet: edge_id, source_tx_id, target_tx_id, source_idx, target_idx, time_step
  scaler_stats.json, validation_report.json, manifest.json

Semantics match the legacy pandas preprocessor (src/data/elliptic.py) so that the
parity check can compare value-by-value: label 1 = illicit, 0 = licit, -1 = unknown;
node order = (time_step, tx_id); edge order = CSV row order.
Splits: train = steps 1-29, val = 30-34, test = 35-49 for labelled nodes; unknown-label
nodes get split = "unlabelled" (kept for graph structure, excluded from loss/metrics).
"""
import hashlib
import json
import os
import shutil
import time
import uuid
from typing import Dict

N_FEATURES = 165
FEATURES = [f"f{i}" for i in range(1, N_FEATURES + 1)]
TRAIN_MAX, VAL_MAX = 29, 34


class EllipticValidationError(RuntimeError):
    pass


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(spark, raw_dir: str, out_dir: str, overwrite: bool = False,
        expected_time_steps=(1, 49, 49)) -> Dict:
    """expected_time_steps = (min, max, n_distinct) or None to skip (used by small test fixtures)."""
    import pyspark.sql.functions as F
    from pyspark.sql import Window
    from pyspark.sql.types import (DoubleType, FloatType, IntegerType, LongType, StringType, StructField,
                                   StructType)
    from src.spark.session import to_spark_path

    t0 = time.time()
    paths = {k: os.path.join(raw_dir, f"elliptic_txs_{k}.csv") for k in ("features", "classes", "edgelist")}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Elliptic {k} file not found: {p}")

    out_abs = os.path.abspath(out_dir)
    if os.path.exists(out_abs) and not overwrite:
        raise FileExistsError(f"{out_abs} exists; pass overwrite=True (the old output is moved to a backup first)")
    staging = os.path.join(os.path.dirname(out_abs), f".staging_elliptic_{uuid.uuid4().hex[:8]}")
    os.makedirs(staging)

    feat_schema = StructType([StructField("tx_id", LongType(), False), StructField("time_step", IntegerType(), False)]
                             + [StructField(c, DoubleType(), True) for c in FEATURES])
    cls_schema = StructType([StructField("tx_id", LongType(), True), StructField("class", StringType(), True)])
    edge_schema = StructType([StructField("source_tx_id", LongType(), True),
                              StructField("target_tx_id", LongType(), True)])

    feats = spark.read.csv(to_spark_path(paths["features"]), schema=feat_schema, header=False, mode="FAILFAST")
    classes = spark.read.csv(to_spark_path(paths["classes"]), schema=cls_schema, header=True, mode="FAILFAST")
    edges_raw = spark.read.csv(to_spark_path(paths["edgelist"]), schema=edge_schema, header=True, mode="FAILFAST")

    report = {"inputs": {k: {"path": p, "bytes": os.path.getsize(p), "sha256": _sha256(p)} for k, p in paths.items()}}
    errors = []

    # ---------------------------------------------------------------- validation
    n_feat = feats.count()
    n_cls = classes.count()
    n_edge = edges_raw.count()
    report["raw_counts"] = {"features": n_feat, "classes": n_cls, "edges": n_edge}

    dup_feat = n_feat - feats.select("tx_id").distinct().count()
    dup_cls = n_cls - classes.select("tx_id").distinct().count()
    dup_edges = n_edge - edges_raw.distinct().count()
    null_ids = (feats.filter(F.col("tx_id").isNull() | F.col("time_step").isNull()).count()
                + classes.filter(F.col("tx_id").isNull()).count()
                + edges_raw.filter(F.col("source_tx_id").isNull() | F.col("target_tx_id").isNull()).count())
    nonfinite_expr = sum(F.when(F.col(c).isNull() | F.isnan(c) | (F.abs(F.col(c)) == float("inf")), 1).otherwise(0)
                         for c in FEATURES)
    n_nonfinite = feats.select(F.sum(nonfinite_expr).alias("n")).first()["n"] or 0
    ts_stats = feats.agg(F.min("time_step").alias("lo"), F.max("time_step").alias("hi"),
                         F.countDistinct("time_step").alias("n")).first()
    cls_norm = classes.withColumn("class", F.lower(F.trim("class")))
    bad_labels = cls_norm.filter(~F.col("class").isin("1", "2", "unknown") | F.col("class").isNull()).count()
    report["integrity"] = {
        "duplicate_tx_ids_features": dup_feat, "duplicate_tx_ids_classes": dup_cls, "duplicate_edges": dup_edges,
        "null_ids": null_ids, "non_finite_feature_values": int(n_nonfinite), "invalid_labels": bad_labels,
        "time_step_min": ts_stats["lo"], "time_step_max": ts_stats["hi"], "time_step_distinct": ts_stats["n"],
    }
    if dup_feat or dup_cls or null_ids or bad_labels:
        errors.append("duplicate/null ids or invalid labels")
    if n_nonfinite:
        errors.append(f"{n_nonfinite} non-finite feature values (legacy pipeline imputes; Spark pipeline refuses)")
    if expected_time_steps and (ts_stats["lo"], ts_stats["hi"], ts_stats["n"]) != tuple(expected_time_steps):
        errors.append(f"unexpected time-step range {ts_stats['lo']}..{ts_stats['hi']} ({ts_stats['n']} distinct)")

    # ---------------------------------------------------------------- alignment
    labelled = cls_norm.withColumn(
        "label", F.when(F.col("class") == "1", 1).when(F.col("class") == "2", 0).otherwise(-1).cast(IntegerType()))
    unmatched_feat = feats.join(labelled, "tx_id", "left_anti").count()
    unmatched_cls = labelled.join(feats, "tx_id", "left_anti").count()
    report["integrity"].update({"unmatched_feature_ids": unmatched_feat, "unmatched_class_ids": unmatched_cls})
    if unmatched_feat or unmatched_cls:
        errors.append("feature/class id mismatch")

    w = Window.orderBy("time_step", "tx_id")
    nodes = (feats.join(labelled.select("tx_id", "label"), "tx_id", "inner")
             .withColumn("node_idx", (F.row_number().over(w) - 1).cast(LongType()))
             .withColumn("label_status", F.when(F.col("label") == -1, "unknown").otherwise("labelled"))
             .withColumn("split",
                         F.when(F.col("label") == -1, "unlabelled")
                         .when(F.col("time_step") <= TRAIN_MAX, "train")
                         .when(F.col("time_step") <= VAL_MAX, "val").otherwise("test")))
    for c in FEATURES:
        nodes = nodes.withColumn(c, F.col(c).cast(FloatType()))  # legacy arrays are float32
    nodes = nodes.select("node_idx", "tx_id", "time_step", "label", "label_status", "split", *FEATURES).cache()
    n_nodes = nodes.count()

    # ---------------------------------------------------------------- edges
    # File-order edge ids computed inside the JVM (Python RDD workers crash on this host).
    # For a file source, monotonically_increasing_id = (partition << 33) + row offset and
    # partitions are byte ranges in file order, so ordering by it reproduces CSV row order.
    edges_idx = (edges_raw.withColumn("_mono", F.monotonically_increasing_id())
                 .withColumn("edge_id", (F.row_number().over(Window.orderBy("_mono")) - 1).cast(LongType()))
                 .drop("_mono"))
    src = nodes.select(F.col("tx_id").alias("source_tx_id"), F.col("node_idx").alias("source_idx"),
                       F.col("time_step").alias("source_step"))
    dst = nodes.select(F.col("tx_id").alias("target_tx_id"), F.col("node_idx").alias("target_idx"),
                       F.col("time_step").alias("target_step"))
    edges = edges_idx.join(src, "source_tx_id", "left").join(dst, "target_tx_id", "left").cache()
    dangling = edges.filter(F.col("source_idx").isNull() | F.col("target_idx").isNull()).count()
    cross = edges.filter(F.col("source_step") != F.col("target_step")).count()
    self_loops = edges.filter(F.col("source_tx_id") == F.col("target_tx_id")).count()
    report["integrity"].update({"dangling_edge_endpoints": dangling, "cross_time_step_edges": cross,
                                "self_loops": self_loops})
    if dangling:
        errors.append(f"{dangling} edges reference unknown transactions")
    if cross:
        errors.append(f"{cross} edges cross time steps (breaks per-step graph isolation)")
    edges = (edges.filter(F.col("source_idx").isNotNull() & F.col("target_idx").isNotNull())
             .select("edge_id", "source_tx_id", "target_tx_id", "source_idx", "target_idx",
                     F.col("source_step").alias("time_step")))

    if errors:
        shutil.rmtree(staging, ignore_errors=True)
        raise EllipticValidationError("; ".join(errors) + f" | report={json.dumps(report['integrity'])}")

    # ---------------------------------------------------------------- split stats
    split_counts = {r["split"]: {"n": r["n"], "illicit": r["illicit"]} for r in
                    nodes.groupBy("split").agg(F.count("*").alias("n"),
                                               F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).alias("illicit"))
                    .collect()}
    report["splits"] = split_counts
    report["labels"] = {str(r["label"]): r["n"] for r in nodes.groupBy("label").agg(F.count("*").alias("n")).collect()}

    # ---------------------------------------------------------------- train-only scaler
    train = nodes.filter(F.col("split") == "train")
    stats_row = train.agg(*[F.avg(F.col(c).cast(DoubleType())).alias(f"m_{c}") for c in FEATURES],
                          *[F.stddev_pop(F.col(c).cast(DoubleType())).alias(f"s_{c}") for c in FEATURES]).first()
    means = [float(stats_row[f"m_{c}"]) for c in FEATURES]
    stds = [float(stats_row[f"s_{c}"]) for c in FEATURES]
    scales = [s if s > 0 else 1.0 for s in stds]  # sklearn convention for zero variance
    scaled = nodes.select("node_idx", "tx_id", "time_step", "label", "split",
                          *[((F.col(c).cast(DoubleType()) - F.lit(m)) / F.lit(s)).cast(FloatType()).alias(c)
                            for c, m, s in zip(FEATURES, means, scales)])

    # ---------------------------------------------------------------- write
    nodes.repartition("time_step").write.mode("overwrite").partitionBy("time_step").parquet(
        to_spark_path(os.path.join(staging, "nodes")))
    scaled.repartition("time_step").write.mode("overwrite").partitionBy("time_step").parquet(
        to_spark_path(os.path.join(staging, "nodes_scaled")))
    edges.orderBy("edge_id").coalesce(1).write.mode("overwrite").parquet(to_spark_path(os.path.join(staging, "edges")))
    n_edges_out = spark.read.parquet(to_spark_path(os.path.join(staging, "edges"))).count()
    n_nodes_out = spark.read.parquet(to_spark_path(os.path.join(staging, "nodes"))).count()
    if n_edges_out != n_edge - dangling or n_nodes_out != n_nodes:
        shutil.rmtree(staging, ignore_errors=True)
        raise EllipticValidationError(f"read-back mismatch: nodes {n_nodes_out}/{n_nodes}, edges {n_edges_out}")

    with open(os.path.join(staging, "scaler_stats.json"), "w") as f:
        json.dump({"fit_on": "split == 'train' (labelled, steps 1-29)", "features": FEATURES, "mean": means,
                   "std_population": stds, "scale_used": scales}, f)
    elapsed = time.time() - t0
    manifest = {
        "dataset": "elliptic", "producer": "scripts/process_elliptic_spark.py", "spark_version": spark.version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_seconds": round(elapsed, 2),
        "n_nodes": n_nodes, "n_edges": n_edges_out, "split_definition": {"train": "1-29", "val": "30-34",
                                                                          "test": "35-49"},
        "label_encoding": {"1": "illicit -> 1", "2": "licit -> 0", "unknown": "-1"},
    }
    report["elapsed_seconds"] = round(elapsed, 2)
    for name, obj in (("validation_report.json", report), ("manifest.json", manifest)):
        with open(os.path.join(staging, name), "w") as f:
            json.dump(obj, f, indent=2)

    backup = None
    if os.path.exists(out_abs):
        backup = f"{out_abs}.backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(out_abs, backup)
    shutil.move(staging, out_abs)
    nodes.unpersist()
    edges.unpersist()
    return {"manifest": manifest, "report": report, "backup": backup}
