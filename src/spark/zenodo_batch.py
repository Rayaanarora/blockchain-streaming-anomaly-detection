"""
Apache Spark Batch Preprocessing Pipeline for Large-Scale Zenodo Bitcoin Transactions.

Transforms raw transaction records into leakage-safe chronological feature tables
and partitioned Snappy-compressed Parquet files.

Fails immediately with an explicit actionable message if PySpark is not installed.
"""
import os
import sys
import argparse
import json
import time
from typing import Dict, Any, List

# Set default project temp and scratch directories strictly on Drive D:
DEFAULT_SPARK_TMP_DIR = "D:/blockchainbigdata/blockchain-streaming-anomaly-detection/data/spark_temp"
DEFAULT_SPARK_WAREHOUSE_DIR = "D:/blockchainbigdata/blockchain-streaming-anomaly-detection/data/spark_warehouse"

# Ensure environment variables default to Drive D: to protect Drive C:
os.makedirs(DEFAULT_SPARK_TMP_DIR, exist_ok=True)
os.makedirs(DEFAULT_SPARK_WAREHOUSE_DIR, exist_ok=True)
os.environ["TEMP"] = DEFAULT_SPARK_TMP_DIR
os.environ["TMP"] = DEFAULT_SPARK_TMP_DIR
os.environ["SPARK_LOCAL_DIRS"] = DEFAULT_SPARK_TMP_DIR

# Ensure JAVA_HOME points to a valid Java runtime
if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ.get("JAVA_HOME", "")):
    if os.path.exists(r"C:\Program Files\Java\jre1.8.0_451"):
        os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jre1.8.0_451"

# Check PySpark availability
try:
    from pyspark.sql import SparkSession, Window
    import pyspark.sql.functions as F
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        LongType,
        IntegerType,
        DoubleType,
        BooleanType
    )
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False


def check_pyspark_available():
    """Fail fast with actionable error message if PySpark is absent."""
    if not HAS_PYSPARK:
        raise RuntimeError(
            "\n" + "=" * 80 + "\n"
            "ERROR: PySpark is not installed in the current environment.\n"
            "This module requires Apache Spark to execute distributed batch transformations.\n"
            "To enable Spark execution, install PySpark:\n"
            "    pip install pyspark==3.5.0\n"
            "and ensure a compatible Java runtime (e.g. Java 8 JRE at C:\\Program Files\\Java\\jre1.8.0_451)\n"
            "is configured in your environment.\n"
            + "=" * 80
        )


def build_spark_session(
    app_name: str = "ZenodoBitcoinBatchPreprocessing",
    local_cores: str = "local[*]",
    spark_tmp_dir: str = DEFAULT_SPARK_TMP_DIR,
    spark_warehouse_dir: str = DEFAULT_SPARK_WAREHOUSE_DIR
) -> "SparkSession":
    """
    Build a local or cluster SparkSession configured specifically for Drive D:
    scratch, warehouse, and checkpoint storage (protecting Drive C: from disk pressure).
    """
    check_pyspark_available()

    # Ensure temp/scratch directories are rooted strictly on Drive D:
    os.makedirs(spark_tmp_dir, exist_ok=True)
    os.makedirs(spark_warehouse_dir, exist_ok=True)
    os.environ["TEMP"] = spark_tmp_dir
    os.environ["TMP"] = spark_tmp_dir
    os.environ["SPARK_LOCAL_DIRS"] = spark_tmp_dir
    os.environ["_JAVA_OPTIONS"] = f"-Djava.io.tmpdir={spark_tmp_dir}"

    # JVM options required for modern Java compatibility and scratch redirection
    jvm_opens = (
        f"-Djava.io.tmpdir={spark_tmp_dir} "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    )

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(local_cores)
        .config("spark.local.dir", spark_tmp_dir)
        .config("spark.sql.warehouse.dir", spark_warehouse_dir)
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.extraJavaOptions", jvm_opens)
        .config("spark.executor.extraJavaOptions", jvm_opens)
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.io.native.lib.available", "false")
    )

    return builder.getOrCreate()


class ZenodoSparkBatchPipeline:
    """
    Distributed batch feature extraction and Parquet writer engine.
    """
    def __init__(self, spark: "SparkSession"):
        check_pyspark_available()
        self.spark = spark

    def parse_raw_zenodo_text(self, text_df: "DataFrame") -> "DataFrame":
        """
        Parse raw space-delimited Zenodo lines into typed, sanitized records.
        Filter malformed lines and non-hex transaction hashes.
        """
        # Split line by whitespace: TYPE BLOCK_NUMBER TX_INDEX FROM TO VALUE
        split_col = F.split(F.trim(text_df["value"]), r"\s+")

        parsed_df = (
            text_df
            .filter(F.size(split_col) == 6)
            .select(
                split_col.getItem(0).alias("kind"),
                split_col.getItem(1).cast("long").alias("block_number"),
                split_col.getItem(2).cast("int").alias("tx_index"),
                split_col.getItem(3).alias("source"),
                split_col.getItem(4).alias("destination"),
                split_col.getItem(5).cast("long").alias("value_satoshi")
            )
            .filter(F.col("kind").isin("BTC-IN", "BTC-OUT"))
            .filter(F.col("block_number").isNotNull() & (F.col("block_number") >= 0))
            .filter(F.col("tx_index").isNotNull() & (F.col("tx_index") >= 0))
            .filter(F.col("value_satoshi").isNotNull() & (F.col("value_satoshi") >= 0))
        )

        # Separate address from tx_hash based on kind:
        # In BTC-IN: source = address, destination = tx_hash
        # In BTC-OUT: source = tx_hash, destination = address
        sanitized_df = (
            parsed_df
            .withColumn(
                "tx_hash",
                F.when(F.col("kind") == "BTC-IN", F.col("destination")).otherwise(F.col("source"))
            )
            .withColumn(
                "raw_address",
                F.when(F.col("kind") == "BTC-IN", F.col("source")).otherwise(F.col("destination"))
            )
            # Filter non-hex tx hashes (must be 64 characters lowercase hex)
            .filter(F.col("tx_hash").rlike("^[0-9a-f]{64}$"))
            # Sanitize unableToDecode and 0 pseudo-addresses to prevent supernode aggregation
            .withColumn(
                "address",
                F.when(
                    F.col("raw_address") == "unableToDecode",
                    F.concat(F.lit("unableToDecode_"), F.substring(F.col("tx_hash"), 1, 16))
                ).when(
                    F.col("raw_address") == "0",
                    F.concat(F.lit("coinbase_source_"), F.substring(F.col("tx_hash"), 1, 16))
                ).otherwise(F.col("raw_address"))
            )
        )

        return sanitized_df

    def assemble_completed_transactions(self, records_df: "DataFrame") -> "DataFrame":
        """
        Group records by (block_number, tx_index, tx_hash) to compute transaction-level totals.
        Coinbase transactions (tx_index == 0) are handled separately with fee = 0.
        """
        tx_agg = (
            records_df
            .groupBy("block_number", "tx_index", "tx_hash")
            .agg(
                F.sum(F.when(F.col("kind") == "BTC-IN", F.col("value_satoshi")).otherwise(0)).alias("total_input_satoshi"),
                F.sum(F.when(F.col("kind") == "BTC-OUT", F.col("value_satoshi")).otherwise(0)).alias("total_output_satoshi"),
                F.sum(F.when(F.col("kind") == "BTC-IN", 1).otherwise(0)).alias("num_inputs"),
                F.sum(F.when(F.col("kind") == "BTC-OUT", 1).otherwise(0)).alias("num_outputs"),
                # Collect input addresses as an array for downstream feature joins
                F.collect_list(F.when(F.col("kind") == "BTC-IN", F.col("address"))).alias("input_addresses")
            )
            .withColumn("is_coinbase", F.col("tx_index") == 0)
            # Filter incomplete standard transactions (must have >= 1 input and >= 1 output)
            .filter(
                (F.col("is_coinbase") & (F.col("num_outputs") > 0)) |
                (~F.col("is_coinbase") & (F.col("num_inputs") > 0) & (F.col("num_outputs") > 0))
            )
            # Fee calculation: coinbase mints new subsidy, fee = 0; standard fee = in - out
            .withColumn(
                "fee_satoshi",
                F.when(F.col("is_coinbase"), 0).otherwise(F.col("total_input_satoshi") - F.col("total_output_satoshi"))
            )
            # Partitioning key for Parquet: 1000-block chunks (e.g. 480000 -> "480000-480999")
            .withColumn(
                "block_chunk",
                F.concat(
                    (F.col("block_number") - (F.col("block_number") % 1000)).cast("string"),
                    F.lit("-"),
                    (F.col("block_number") - (F.col("block_number") % 1000) + 999).cast("string")
                )
            )
        )

        return tx_agg

    def compute_leakage_safe_features(self, records_df: "DataFrame", tx_df: "DataFrame") -> "DataFrame":
        """
        Compute chronological graph and volume features using Spark Window functions.
        STRICT TEMPORAL CAUSALITY: Window upper bound is rowsBetween(unboundedPreceding, -1),
        guaranteeing that historical counts use ONLY transactions strictly prior to the current (block, tx_index).

        Ensures there is at most one address-history row per (address, block_number, tx_index)
        prior to window evaluation. Prevents intra-transaction leakage (e.g. change outputs
        being counted as prior history for inputs of the same transaction).
        """
        # Address-level participation events: aggregate all UTXOs per (address, block_number, tx_index)
        # into exactly one row per transaction participation.
        addr_events = (
            records_df
            .groupBy("address", "block_number", "tx_index")
            .agg(
                F.sum(F.when(F.col("kind") == "BTC-IN", F.col("value_satoshi")).otherwise(0)).alias("in_satoshi"),
                F.sum(F.when(F.col("kind") == "BTC-OUT", F.col("value_satoshi")).otherwise(0)).alias("out_satoshi"),
                F.max(F.when(F.col("kind") == "BTC-IN", 1).otherwise(0)).alias("has_input")
            )
            # Representative volume per transaction: max(in_satoshi, out_satoshi)
            # avoids double-counting change outputs while capturing true transferred volume.
            .withColumn("tx_volume_satoshi", F.greatest(F.col("in_satoshi"), F.col("out_satoshi")))
        )

        # Window partitioned by address, sorted strictly chronologically.
        # Since (address, block_number, tx_index) is unique, the ordering is strictly deterministic.
        # Upper bound -1 excludes the current transaction and any future transactions!
        addr_hist_window = (
            Window.partitionBy("address")
            .orderBy("block_number", "tx_index")
            .rowsBetween(Window.unboundedPreceding, -1)
        )

        addr_with_history = (
            addr_events
            .withColumn("prior_tx_count", F.coalesce(F.count(F.lit(1)).over(addr_hist_window), F.lit(0)))
            .withColumn("prior_volume_satoshi", F.coalesce(F.sum("tx_volume_satoshi").over(addr_hist_window), F.lit(0)))
            .filter(F.col("has_input") == 1)  # Focus on input addresses
            .select("address", "block_number", "tx_index", "prior_tx_count", "prior_volume_satoshi")
        )

        # Join with each input record to preserve input-level weighting
        # (matching reference research extractor.py: avg_input_prior_tx_count and new_address_fraction
        # are computed over all inputs of the transaction).
        inputs_with_history = (
            records_df
            .filter(F.col("kind") == "BTC-IN")
            .select("address", "block_number", "tx_index")
            .join(addr_with_history, on=["address", "block_number", "tx_index"], how="left")
            .withColumn("prior_tx_count", F.coalesce(F.col("prior_tx_count"), F.lit(0)))
            .withColumn("prior_volume_satoshi", F.coalesce(F.col("prior_volume_satoshi"), F.lit(0)))
        )

        # Aggregate inputs back to transaction level
        tx_hist_stats = (
            inputs_with_history
            .groupBy("block_number", "tx_index")
            .agg(
                F.max("prior_tx_count").alias("max_input_prior_tx_count"),
                F.avg("prior_tx_count").alias("avg_input_prior_tx_count"),
                F.max("prior_volume_satoshi").alias("max_input_prior_volume_satoshi"),
                (F.sum(F.when(F.col("prior_tx_count") == 0, 1).otherwise(0)) / F.count(F.lit(1))).alias("new_address_fraction")
            )
        )

        # Join transaction totals with historical features
        features_df = (
            tx_df.join(tx_hist_stats, on=["block_number", "tx_index"], how="left")
            .withColumn("max_input_prior_tx_count", F.coalesce(F.col("max_input_prior_tx_count"), F.lit(0)))
            .withColumn("avg_input_prior_tx_count", F.coalesce(F.col("avg_input_prior_tx_count"), F.lit(0.0)))
            .withColumn("max_input_prior_volume_satoshi", F.coalesce(F.col("max_input_prior_volume_satoshi"), F.lit(0)))
            .withColumn(
                "new_address_fraction",
                F.when(F.col("is_coinbase"), 0.0).otherwise(F.coalesce(F.col("new_address_fraction"), F.lit(1.0)))
            )
            # Local ratio features
            .withColumn("fee_rate", F.col("fee_satoshi") / (F.col("num_inputs") + F.col("num_outputs")))
            .withColumn("avg_input_val", F.col("total_input_satoshi") / F.col("num_inputs"))
            .withColumn("avg_output_val", F.col("total_output_satoshi") / F.col("num_outputs"))
            .withColumn("input_output_count_ratio", F.col("num_inputs").cast("double") / F.col("num_outputs").cast("double"))
            .withColumn(
                "input_output_val_ratio",
                F.when(F.col("total_output_satoshi") > 0, F.col("total_input_satoshi").cast("double") / F.col("total_output_satoshi").cast("double")).otherwise(1.0)
            )
            .withColumn("is_coinbase", F.when(F.col("is_coinbase"), 1).otherwise(0))
            .drop("input_addresses")
        )

        return features_df

    def write_parquet_bounded_arrow(
        self,
        features_df: "DataFrame",
        output_parquet_dir: str,
        batch_size: int = 1000
    ) -> None:
        """
        Write Snappy-compressed partitioned Parquet files using a bounded PyArrow iterator.

        Architectural Guarantees & Memory Bounds:
        1. Never uses df.collect() or df.toPandas() on the entire dataset.
        2. Iterates partition by partition (chunk by chunk), streaming at most `batch_size` rows (default 1,000)
           in Python memory per batch (~1 MB buffer).
        3. Memory Bounds:
           - toLocalIterator() yields one partition at a time to the driver via socket streaming, preventing
             full-dataset materialization. However, Spark's driver process maintains JVM internal buffers
             and PySpark communication sockets, typically resulting in an operating RSS of ~100-150 MB.
           - Catalyst distributed sorting (orderBy block_number, tx_index) executes external merge-sort
             across worker executors, spilling to Drive D: scratch if executor memory thresholds are reached.
        4. Safe Two-Phase Directory Replacement:
           - Writes atomically into a staging directory on Drive D: (.staging_{uuid}).
           - If destination already exists, moves destination to a temporary backup directory (.backup_{uuid})
             BEFORE moving the verified staging directory into place.
           - If replacement fails at any point, automatically rolls back from the backup directory, guaranteeing
             that a failed replacement can NEVER destroy or corrupt the previous valid output.
        5. Preserves exact block_chunk partition structure: output_dir/block_chunk={chunk}/part-0.parquet.
        """
        import uuid
        import shutil
        import pyarrow as pa
        import pyarrow.parquet as pq
        from src.spark.schemas import get_transaction_features_pyarrow_schema

        arrow_schema = get_transaction_features_pyarrow_schema()
        output_abs = os.path.abspath(output_parquet_dir)
        parent_dir = os.path.dirname(output_abs)
        os.makedirs(parent_dir, exist_ok=True)

        # Staging directory strictly on Drive D: for atomic commit
        staging_dir = os.path.join(parent_dir, f".staging_{uuid.uuid4().hex[:8]}")
        backup_dir = None
        os.makedirs(staging_dir, exist_ok=True)

        try:
            # 1. Discover distinct block_chunk partition keys (bounded metadata stream, zero full collection)
            chunk_keys = [
                r[0] for r in (
                    features_df
                    .select("block_chunk")
                    .distinct()
                    .orderBy("block_chunk")
                    .toLocalIterator()
                )
            ]

            # Feature column names matching arrow_schema
            feature_cols = [f.name for f in arrow_schema]

            # 2. Process partition by partition
            for chunk_key in chunk_keys:
                chunk_dir = os.path.join(staging_dir, f"block_chunk={chunk_key}")
                os.makedirs(chunk_dir, exist_ok=True)
                part_file = os.path.join(chunk_dir, "part-0.parquet")

                # Filter strictly to this chunk and order deterministically
                chunk_df = (
                    features_df
                    .filter(F.col("block_chunk") == chunk_key)
                    .select(feature_cols)
                    .orderBy("block_number", "tx_index")
                )

                # Initialize streaming Parquet writer
                writer = pq.ParquetWriter(
                    part_file,
                    arrow_schema,
                    compression="snappy"
                )

                try:
                    batch_buffer = []
                    # Stream rows using toLocalIterator (bounded memory, never buffers full partition)
                    for row in chunk_df.toLocalIterator():
                        row_dict = row.asDict()
                        batch_buffer.append(row_dict)
                        if len(batch_buffer) >= batch_size:
                            record_batch = pa.RecordBatch.from_pylist(batch_buffer, schema=arrow_schema)
                            writer.write_batch(record_batch)
                            batch_buffer.clear()

                    # Write residual rows in buffer
                    if batch_buffer:
                        record_batch = pa.RecordBatch.from_pylist(batch_buffer, schema=arrow_schema)
                        writer.write_batch(record_batch)
                        batch_buffer.clear()
                finally:
                    writer.close()

            # Write standard Parquet _SUCCESS marker
            with open(os.path.join(staging_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
                f.write("")

            # 3. Two-Phase Safe Directory Replacement:
            # Step A: If target exists, move it to a temporary backup directory first.
            if os.path.exists(output_abs):
                backup_dir = os.path.join(parent_dir, f".backup_{uuid.uuid4().hex[:8]}")
                shutil.move(output_abs, backup_dir)

            try:
                # Step B: Move fully staged and verified directory to target path
                shutil.move(staging_dir, output_abs)
                # Step C: Success! Clean up backup directory
                if backup_dir and os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir, ignore_errors=True)
            except Exception as replace_err:
                # Rollback! Restore original target from backup directory
                if backup_dir and os.path.exists(backup_dir):
                    if os.path.exists(output_abs):
                        shutil.rmtree(output_abs, ignore_errors=True)
                    shutil.move(backup_dir, output_abs)
                raise replace_err

        except Exception as e:
            # Clean up temporary staging directory on any failure
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise e

    def run_pipeline(
        self,
        input_paths: List[str],
        output_parquet_dir: str,
        audit_report_path: str
    ) -> Dict[str, Any]:
        """
        Execute end-to-end Spark batch transformation and write Snappy Parquet via bounded PyArrow.
        """
        t0 = time.time()
        print(f"Reading raw Zenodo records from: {input_paths}")
        raw_text_df = self.spark.read.text(input_paths)
        raw_count = raw_text_df.count()

        print("Parsing and sanitizing records...")
        records_df = self.parse_raw_zenodo_text(raw_text_df)
        parsed_count = records_df.count()

        print("Assembling completed transactions...")
        tx_df = self.assemble_completed_transactions(records_df)
        tx_count = tx_df.count()

        print("Computing chronological graph features (windowed strictly prior)...")
        features_df = self.compute_leakage_safe_features(records_df, tx_df)

        print(f"Writing Snappy-compressed Parquet via bounded PyArrow to: {output_parquet_dir}")
        self.write_parquet_bounded_arrow(features_df, output_parquet_dir)

        elapsed = time.time() - t0
        audit_report = {
            "execution_mode": "Apache Spark Batch + Bounded PyArrow Parquet Writer",
            "spark_version": self.spark.version,
            "raw_lines_read": raw_count,
            "valid_records_parsed": parsed_count,
            "completed_transactions": tx_count,
            "output_parquet_directory": output_parquet_dir,
            "compression": "snappy",
            "elapsed_seconds": round(elapsed, 2),
            "throughput_lines_per_sec": round(raw_count / max(elapsed, 1e-6), 2)
        }

        os.makedirs(os.path.dirname(os.path.abspath(audit_report_path)), exist_ok=True)
        with open(audit_report_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)

        print(f"Audit report saved to: {audit_report_path}")
        return audit_report


def main():
    parser = argparse.ArgumentParser(description="Apache Spark Batch Preprocessing for Zenodo Transactions.")
    parser.add_argument("--input-paths", nargs="+", required=True, help="Input text or archive paths.")
    parser.add_argument("--output-parquet-dir", default="data/parquet/zenodo_features", help="Output Parquet directory.")
    parser.add_argument("--audit-report-path", default="data/validation/spark_batch_audit.json", help="Audit report path.")
    parser.add_argument("--spark-tmp-dir", default="D:/blockchainbigdata/blockchain-streaming-anomaly-detection/data/spark_temp", help="Spark scratch dir (must be on D:).")
    parser.add_argument("--spark-warehouse-dir", default="D:/blockchainbigdata/blockchain-streaming-anomaly-detection/data/spark_warehouse", help="Spark warehouse dir (must be on D:).")
    parser.add_argument("--master", default="local[*]", help="Spark master URL (default: local[*]).")
    args = parser.parse_args()

    # Verify PySpark availability
    check_pyspark_available()

    spark = build_spark_session(
        local_cores=args.master,
        spark_tmp_dir=args.spark_tmp_dir,
        spark_warehouse_dir=args.spark_warehouse_dir
    )
    try:
        pipeline = ZenodoSparkBatchPipeline(spark)
        report = pipeline.run_pipeline(args.input_paths, args.output_parquet_dir, args.audit_report_path)
        print("Spark batch preprocessing finished successfully.")
        print(json.dumps(report, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
