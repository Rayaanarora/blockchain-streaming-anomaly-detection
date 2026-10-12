"""
Tests for Apache Spark schemas, validation rules, chronological window logic,
and batch processing specifications.

Tests are partitioned:
1. Environment-agnostic schema contract and causality logic tests (run in any Python env).
2. PySpark integration tests (guarded by pytest.mark.skipif when PySpark is absent).
"""
import pytest
from src.spark.schemas import (
    ZENODO_RAW_RECORD_SPEC,
    COMPLETED_TRANSACTION_SPEC,
    TRANSACTION_FEATURE_SPEC,
    HAS_PYSPARK,
    get_raw_record_spark_schema,
    get_completed_transaction_spark_schema,
    get_transaction_features_spark_schema
)
from src.spark.zenodo_batch import check_pyspark_available


# ==============================================================================
# 1. Platform-Agnostic Schema Contract Tests
# ==============================================================================

def test_zenodo_raw_record_schema_spec():
    """Verify raw Zenodo schema specification defines all required fields with correct types."""
    expected_fields = {
        "kind": "string",
        "block_number": "long",
        "tx_index": "integer",
        "source": "string",
        "destination": "string",
        "value_satoshi": "long"
    }
    assert ZENODO_RAW_RECORD_SPEC == expected_fields


def test_completed_transaction_schema_spec():
    """Verify completed transaction schema includes partition key and fee."""
    assert "fee_satoshi" in COMPLETED_TRANSACTION_SPEC
    assert "is_coinbase" in COMPLETED_TRANSACTION_SPEC
    assert "block_chunk" in COMPLETED_TRANSACTION_SPEC
    assert COMPLETED_TRANSACTION_SPEC["block_number"] == "long"
    assert COMPLETED_TRANSACTION_SPEC["tx_index"] == "integer"


def test_transaction_feature_schema_spec():
    """Verify 15-feature contract matches historical feature extractor definitions."""
    required_feature_fields = [
        "fee_rate",
        "avg_input_val",
        "avg_output_val",
        "input_output_count_ratio",
        "input_output_val_ratio",
        "max_input_prior_tx_count",
        "avg_input_prior_tx_count",
        "max_input_prior_volume_satoshi",
        "new_address_fraction"
    ]
    for feat in required_feature_fields:
        assert feat in TRANSACTION_FEATURE_SPEC


def test_spark_absent_error_actionability():
    """Verify check_pyspark_available raises actionable message if PySpark is absent."""
    if not HAS_PYSPARK:
        with pytest.raises(RuntimeError, match="PySpark is not installed in the current environment"):
            check_pyspark_available()

        with pytest.raises(ImportError, match="Install pyspark"):
            get_raw_record_spark_schema()
    else:
        # If PySpark is installed, ensure schema instantiates properly
        schema = get_raw_record_spark_schema()
        assert len(schema.fields) == 6


# ==============================================================================
# 2. Chronological Causality Logic Test (Simulated Window Math)
# ==============================================================================

def test_chronological_window_causality_math():
    """
    Verify that an upper bound of -1 in a window frame (unboundedPreceding to -1)
    mathematically guarantees zero leakage of current or future rows.
    """
    # Simulate a chronological series of transactions for address A
    tx_series = [
        {"block": 100, "tx_idx": 1, "val": 1000},
        {"block": 100, "tx_idx": 2, "val": 2500},
        {"block": 101, "tx_idx": 0, "val": 5000},
    ]

    # Emulate Spark rowsBetween(unboundedPreceding, -1)
    prior_counts = []
    prior_volumes = []
    for i in range(len(tx_series)):
        # strictly rows 0 .. i-1
        past_rows = tx_series[:i]
        prior_counts.append(len(past_rows))
        prior_volumes.append(sum(r["val"] for r in past_rows))

    # At row 0: prior count must be 0, prior volume 0 (brand new address)
    assert prior_counts[0] == 0
    assert prior_volumes[0] == 0

    # At row 1: prior count must be 1, prior volume 1000 (reflects row 0 only)
    assert prior_counts[1] == 1
    assert prior_volumes[1] == 1000

    # At row 2: prior count must be 2, prior volume 3500 (reflects rows 0 and 1 only)
    assert prior_counts[2] == 2
    assert prior_volumes[2] == 3500


# ==============================================================================
# 3. PySpark Live Execution Integration Tests (Skipped when PySpark absent)
# ==============================================================================

@pytest.mark.skipif(not HAS_PYSPARK, reason="PySpark is not installed in local environment")
def test_pyspark_batch_pipeline_execution():
    """
    End-to-end integration test of Spark batch pipeline using bounded PyArrow writer.
    Directly compares PyArrow Parquet output against small known-correct Spark result.
    Executes on Drive D: scratch directory with zero writes to Drive C:.
    """
    import os
    import shutil
    import pyarrow.parquet as pq
    from src.spark.zenodo_batch import build_spark_session, ZenodoSparkBatchPipeline

    test_scratch_dir = "D:/blockchainbigdata/blockchain-streaming-anomaly-detection/data/test_spark_scratch"
    if os.path.exists(test_scratch_dir):
        shutil.rmtree(test_scratch_dir, ignore_errors=True)
    os.makedirs(test_scratch_dir, exist_ok=True)

    spark = build_spark_session(
        app_name="TestPySparkBatch",
        local_cores="local[2]",
        spark_tmp_dir=f"{test_scratch_dir}/spark_temp",
        spark_warehouse_dir=f"{test_scratch_dir}/spark_warehouse"
    )
    try:
        sample_lines = [
            # 1. Coinbase transaction
            "BTC-IN 480000 0 0 5ade05890effc4adce70fbe97908bb803b10dc4a6247615496b21538f132c0c6 1250000000",
            "BTC-OUT 480000 0 5ade05890effc4adce70fbe97908bb803b10dc4a6247615496b21538f132c0c6 1MinerPool 1250000000",
            # 2. Standard transaction
            "BTC-IN 480000 1 1Alice f1a3b08e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef2 50000",
            "BTC-OUT 480000 1 f1a3b08e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef2 1Bob 45000",
            # 3. Second block transaction with repeated address (temporal window check)
            "BTC-IN 480001 1 1Alice d3c5b28e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef4 20000",
            "BTC-OUT 480001 1 d3c5b28e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef4 1Charlie 18000"
        ]
        sample_file = os.path.join(test_scratch_dir, "sample.txt")
        with open(sample_file, "w", encoding="ascii") as f:
            f.write("\n".join(sample_lines))

        out_parquet = os.path.join(test_scratch_dir, "out_parquet")
        audit_file = os.path.join(test_scratch_dir, "audit.json")

        pipeline = ZenodoSparkBatchPipeline(spark)
        
        # 1. Compute in-memory Spark DataFrame baseline via bounded iterator
        raw_text_df = spark.read.text([sample_file])
        records_df = pipeline.parse_raw_zenodo_text(raw_text_df)
        tx_df = pipeline.assemble_completed_transactions(records_df)
        features_df = pipeline.compute_leakage_safe_features(records_df, tx_df)
        spark_rows = [r.asDict() for r in features_df.toLocalIterator()]
        assert len(spark_rows) == 3

        # 2. Write Parquet via bounded PyArrow writer
        pipeline.write_parquet_bounded_arrow(features_df, out_parquet)

        # 3. Test retry safety: second write replaces output cleanly
        pipeline.write_parquet_bounded_arrow(features_df, out_parquet)

        # 4. Validate output directory structure and _SUCCESS marker
        assert os.path.exists(out_parquet)
        assert os.path.exists(os.path.join(out_parquet, "_SUCCESS")), "Missing _SUCCESS marker"
        chunk_dirs = [d for d in os.listdir(out_parquet) if d.startswith("block_chunk=")]
        assert len(chunk_dirs) == 1
        assert chunk_dirs[0] == "block_chunk=480000-480999"

        # 5. Read back Parquet via PyArrow and compare directly against Spark baseline
        table = pq.read_table(out_parquet)
        pa_rows = table.to_pylist()
        assert len(pa_rows) == len(spark_rows) == 3

        pa_sorted = sorted(pa_rows, key=lambda r: (r["block_number"], r["tx_index"]))
        spark_sorted = sorted(spark_rows, key=lambda r: (r["block_number"], r["tx_index"]))

        for pa_r, sp_r in zip(pa_sorted, spark_sorted):
            assert pa_r["block_number"] == sp_r["block_number"]
            assert pa_r["tx_index"] == sp_r["tx_index"]
            assert pa_r["tx_hash"] == sp_r["tx_hash"]
            assert pa_r["num_inputs"] == sp_r["num_inputs"]
            assert pa_r["num_outputs"] == sp_r["num_outputs"]
            assert pa_r["total_input_satoshi"] == sp_r["total_input_satoshi"]
            assert pa_r["total_output_satoshi"] == sp_r["total_output_satoshi"]
            assert pa_r["fee_satoshi"] == sp_r["fee_satoshi"]
            assert pa_r["is_coinbase"] == sp_r["is_coinbase"]
            assert pa_r["max_input_prior_tx_count"] == sp_r["max_input_prior_tx_count"]
            assert pa_r["max_input_prior_volume_satoshi"] == sp_r["max_input_prior_volume_satoshi"]

            for float_col in [
                "fee_rate", "avg_input_val", "avg_output_val",
                "input_output_count_ratio", "input_output_val_ratio",
                "avg_input_prior_tx_count", "new_address_fraction"
            ]:
                assert abs(pa_r[float_col] - sp_r[float_col]) < 1e-5

        # Check semantic domain rules
        by_key = {(r["block_number"], r["tx_index"]): r for r in pa_rows}
        # Coinbase
        assert by_key[(480000, 0)]["is_coinbase"] == 1
        assert by_key[(480000, 0)]["fee_satoshi"] == 0
        assert by_key[(480000, 0)]["new_address_fraction"] == 0.0
        # Alice 1st tx
        assert by_key[(480000, 1)]["fee_satoshi"] == 5000
        assert by_key[(480000, 1)]["max_input_prior_tx_count"] == 0
        # Alice 2nd tx in next block
        assert by_key[(480001, 1)]["fee_satoshi"] == 2000
        assert by_key[(480001, 1)]["max_input_prior_tx_count"] == 1
        assert by_key[(480001, 1)]["max_input_prior_volume_satoshi"] == 50000

    finally:
        spark.stop()
        if os.path.exists(test_scratch_dir):
            shutil.rmtree(test_scratch_dir, ignore_errors=True)


@pytest.mark.skipif(not HAS_PYSPARK, reason="PySpark is not installed in local environment")
def test_pyspark_temporal_leakage_regression_scenarios():
    """
    Comprehensive regression tests verifying temporal leakage fix:
    1. Address appearing as both input and output in same transaction (change address).
       - Zero prior history for its own transaction; does NOT leak its own change output.
       - Next transaction sees prior_tx_count = 1 (not 2) and volume is not double-counted.
    2. Repeated inputs in same transaction (multi-UTXO consolidation).
       - Prior transaction count increments by 1, not N.
    3. Repeated outputs in same transaction (multi-output).
       - Prior transaction count increments by 1, not N.
    4. Multiple transactions in one block.
       - Deterministic chronological ordering by tx_index within block.
    5. Transactions crossing chunk boundaries (block 480999 -> 481000).
       - Historical features preserve causality across 1000-block Parquet partition boundaries.
    6. First occurrence of every address has strictly zero prior history.
    """
    import os
    import shutil
    import pyarrow.parquet as pq
    from src.spark.zenodo_batch import build_spark_session, ZenodoSparkBatchPipeline

    test_scratch_dir = "D:/blockchainbigdata/blockchain-streaming-anomaly-detection/data/test_spark_causality_reg"
    if os.path.exists(test_scratch_dir):
        shutil.rmtree(test_scratch_dir, ignore_errors=True)
    os.makedirs(test_scratch_dir, exist_ok=True)

    spark = build_spark_session(
        app_name="TestPySparkCausalityRegression",
        local_cores="local[2]",
        spark_tmp_dir=f"{test_scratch_dir}/spark_temp",
        spark_warehouse_dir=f"{test_scratch_dir}/spark_warehouse"
    )

    try:
        # Create deterministic 64-char hex transaction hashes
        h1 = "1111111111111111111111111111111111111111111111111111111111110001"
        h2 = "1111111111111111111111111111111111111111111111111111111111110002"
        h3 = "1111111111111111111111111111111111111111111111111111111111110003"
        h4 = "1111111111111111111111111111111111111111111111111111111111110004"
        h5 = "1111111111111111111111111111111111111111111111111111111111110005"
        h6 = "1111111111111111111111111111111111111111111111111111111111110006"

        sample_lines = [
            # Tx 1: Block 480000, tx 1: 1Alice has both input (100k) and change output (90k)
            f"BTC-IN 480000 1 1Alice {h1} 100000",
            f"BTC-OUT 480000 1 {h1} 1Alice 90000",
            f"BTC-OUT 480000 1 {h1} 1Merchant 8000",

            # Tx 2: Block 480000, tx 2: 1MultiIn has 3 inputs; 1MultiOut has 2 outputs
            f"BTC-IN 480000 2 1MultiIn {h2} 50000",
            f"BTC-IN 480000 2 1MultiIn {h2} 30000",
            f"BTC-IN 480000 2 1MultiIn {h2} 20000",
            f"BTC-OUT 480000 2 {h2} 1MultiOut 40000",
            f"BTC-OUT 480000 2 {h2} 1MultiOut 55000",

            # Tx 3: Block 480000, tx 3: 1Alice spends again in SAME BLOCK
            f"BTC-IN 480000 3 1Alice {h3} 40000",
            f"BTC-OUT 480000 3 {h3} 1Bob 38000",

            # Tx 4: Block 480000, tx 4: 1MultiIn spends again in SAME BLOCK
            f"BTC-IN 480000 4 1MultiIn {h4} 10000",
            f"BTC-OUT 480000 4 {h4} 1Bob 9000",

            # Tx 5: Block 480999, tx 1: 1MultiOut spends in chunk 480000-480999
            f"BTC-IN 480999 1 1MultiOut {h5} 60000",
            f"BTC-OUT 480999 1 {h5} 1David 58000",

            # Tx 6: Block 481000, tx 1: 1MultiOut spends in NEW CHUNK 481000-481999
            f"BTC-IN 481000 1 1MultiOut {h6} 20000",
            f"BTC-OUT 481000 1 {h6} 1David 19000",
        ]

        sample_file = os.path.join(test_scratch_dir, "regression_sample.txt")
        with open(sample_file, "w", encoding="ascii") as f:
            f.write("\n".join(sample_lines))

        out_parquet = os.path.join(test_scratch_dir, "out_parquet")
        pipeline = ZenodoSparkBatchPipeline(spark)

        raw_text_df = spark.read.text([sample_file])
        records_df = pipeline.parse_raw_zenodo_text(raw_text_df)
        tx_df = pipeline.assemble_completed_transactions(records_df)
        features_df = pipeline.compute_leakage_safe_features(records_df, tx_df)

        pipeline.write_parquet_bounded_arrow(features_df, out_parquet)

        # Read back from Parquet across partitions
        table = pq.read_table(out_parquet)
        rows = table.to_pylist()
        by_key = {(r["block_number"], r["tx_index"]): r for r in rows}

        assert len(by_key) == 6

        # --- Scenario 1: Change Address in Tx 1 ---
        # 1Alice appeared as input (100k) and change output (90k).
        # Her first occurrence must have 0 prior txs, 0 prior volume, and new_address_fraction = 1.0
        tx1 = by_key[(480000, 1)]
        assert tx1["num_inputs"] == 1
        assert tx1["num_outputs"] == 2
        assert tx1["fee_satoshi"] == 2000
        assert tx1["max_input_prior_tx_count"] == 0, "Change output leaked into own transaction!"
        assert tx1["avg_input_prior_tx_count"] == 0.0
        assert tx1["max_input_prior_volume_satoshi"] == 0
        assert tx1["new_address_fraction"] == 1.0

        # --- Scenario 2 & 3: Repeated Inputs & Outputs in Tx 2 ---
        # 1MultiIn has 3 input UTXOs. First occurrence: prior_count = 0.
        tx2 = by_key[(480000, 2)]
        assert tx2["num_inputs"] == 3
        assert tx2["num_outputs"] == 2
        assert tx2["fee_satoshi"] == 5000
        assert tx2["max_input_prior_tx_count"] == 0
        assert tx2["max_input_prior_volume_satoshi"] == 0
        assert tx2["new_address_fraction"] == 1.0

        # --- Scenario 4: Multiple Txs in One Block & Change Address Prior Count in Tx 3 ---
        # 1Alice spends again in block 480000, tx 3.
        # Since tx 1 was her only earlier transaction, prior_tx_count MUST be 1 (NOT 2).
        # Prior volume MUST be 100000 (NOT 100000 + 90000 = 190000, no double-counting change).
        tx3 = by_key[(480000, 3)]
        assert tx3["num_inputs"] == 1
        assert tx3["num_outputs"] == 1
        assert tx3["max_input_prior_tx_count"] == 1, "Input and change output counted as separate historical events!"
        assert tx3["avg_input_prior_tx_count"] == 1.0
        assert tx3["max_input_prior_volume_satoshi"] == 100000, "Change output volume double-counted!"
        assert tx3["new_address_fraction"] == 0.0

        # --- Scenario 2 continued: Repeated inputs in Tx 4 ---
        # 1MultiIn spends again in tx 4.
        # Prior transaction count must be 1 (Tx 2), NOT 3 (the 3 input UTXOs from Tx 2).
        tx4 = by_key[(480000, 4)]
        assert tx4["max_input_prior_tx_count"] == 1, "Multi-UTXO consolidation inflated prior transaction count!"
        assert tx4["avg_input_prior_tx_count"] == 1.0
        assert tx4["max_input_prior_volume_satoshi"] == 100000
        assert tx4["new_address_fraction"] == 0.0

        # --- Scenario 3 continued & Chunk boundary 1 in Tx 5 ---
        # 1MultiOut spends in block 480999, tx 1.
        # In Tx 2, 1MultiOut received 2 outputs totaling 95000 sat.
        # Prior transaction count must be 1 (Tx 2), NOT 2 (the 2 output UTXOs).
        # Prior volume must be 95000.
        tx5 = by_key[(480999, 1)]
        assert tx5["block_chunk"] == "480000-480999"
        assert tx5["max_input_prior_tx_count"] == 1, "Repeated outputs inflated prior transaction count!"
        assert tx5["avg_input_prior_tx_count"] == 1.0
        assert tx5["max_input_prior_volume_satoshi"] == 95000
        assert tx5["new_address_fraction"] == 0.0

        # --- Scenario 5: Crossing Chunk Boundary in Tx 6 ---
        # 1MultiOut spends in block 481000, tx 1.
        # Belongs to partition block_chunk=481000-481999.
        # Earlier history includes Tx 2 (vol 95000) and Tx 5 (vol 60000).
        # Prior count must be 2. Prior volume must be 95000 + 60000 = 155000.
        tx6 = by_key[(481000, 1)]
        assert tx6["block_chunk"] == "481000-481999"
        assert tx6["max_input_prior_tx_count"] == 2
        assert tx6["avg_input_prior_tx_count"] == 2.0
        assert tx6["max_input_prior_volume_satoshi"] == 155000
        assert tx6["new_address_fraction"] == 0.0

        # Verify physical Parquet partition layout
        chunk_dirs = sorted([d for d in os.listdir(out_parquet) if d.startswith("block_chunk=")])
        assert chunk_dirs == ["block_chunk=480000-480999", "block_chunk=481000-481999"]

    finally:
        spark.stop()
        if os.path.exists(test_scratch_dir):
            shutil.rmtree(test_scratch_dir, ignore_errors=True)


@pytest.mark.skipif(not HAS_PYSPARK, reason="PySpark is not installed in local environment")
def test_parquet_writer_failure_modes_and_atomic_rollback():
    """
    Test failure modes in write_parquet_bounded_arrow:
    1. Failure during staging: original output is preserved; staging dir is deleted.
    2. Failure during final replacement: two-phase replacement rolls back from backup;
       original output is preserved intact.
    """
    import os
    import shutil
    from unittest import mock
    from src.spark.zenodo_batch import build_spark_session, ZenodoSparkBatchPipeline

    test_scratch_dir = "D:/blockchainbigdata/blockchain-streaming-anomaly-detection/data/test_spark_fail_modes"
    if os.path.exists(test_scratch_dir):
        shutil.rmtree(test_scratch_dir, ignore_errors=True)
    os.makedirs(test_scratch_dir, exist_ok=True)

    spark = build_spark_session(
        app_name="TestPySparkFailureModes",
        local_cores="local[2]",
        spark_tmp_dir=f"{test_scratch_dir}/spark_temp",
        spark_warehouse_dir=f"{test_scratch_dir}/spark_warehouse"
    )

    try:
        pipeline = ZenodoSparkBatchPipeline(spark)
        out_parquet = os.path.join(test_scratch_dir, "valid_parquet_output")

        # 1. Establish an initial valid output directory with a sentinel file
        os.makedirs(out_parquet, exist_ok=True)
        sentinel_path = os.path.join(out_parquet, "sentinel_valid_v1.txt")
        with open(sentinel_path, "w", encoding="utf-8") as f:
            f.write("VALID_DATA_V1")

        # Create a small valid features DataFrame
        sample_lines = [
            "BTC-IN 480000 1 1Alice 1111111111111111111111111111111111111111111111111111111111110001 50000",
            "BTC-OUT 480000 1 1111111111111111111111111111111111111111111111111111111111110001 1Bob 45000"
        ]
        sample_file = os.path.join(test_scratch_dir, "sample.txt")
        with open(sample_file, "w", encoding="ascii") as f:
            f.write("\n".join(sample_lines))
        raw_text_df = spark.read.text([sample_file])
        records_df = pipeline.parse_raw_zenodo_text(raw_text_df)
        tx_df = pipeline.assemble_completed_transactions(records_df)
        features_df = pipeline.compute_leakage_safe_features(records_df, tx_df)

        # -------------------------------------------------------------
        # Scenario 1: Failure during staging (e.g. iterator crashes)
        # -------------------------------------------------------------
        with mock.patch.object(features_df, "select", side_effect=RuntimeError("Simulated staging crash")):
            with pytest.raises(RuntimeError, match="Simulated staging crash"):
                pipeline.write_parquet_bounded_arrow(features_df, out_parquet)

        # Verification 1: Original valid output is completely untouched!
        assert os.path.exists(out_parquet), "Output directory was prematurely deleted during staging failure!"
        assert os.path.exists(sentinel_path), "Sentinel file missing after staging failure!"
        with open(sentinel_path, "r", encoding="utf-8") as f:
            assert f.read() == "VALID_DATA_V1"
        # No orphan staging directories
        staging_dirs = [d for d in os.listdir(test_scratch_dir) if d.startswith(".staging_")]
        assert len(staging_dirs) == 0, f"Orphan staging dir found: {staging_dirs}"

        # -------------------------------------------------------------
        # Scenario 2: Failure during final replacement (move fails, triggers rollback)
        # -------------------------------------------------------------
        real_move = shutil.move
        def failing_move(src, dst, **kwargs):
            if os.path.basename(src).startswith(".staging_") and dst == os.path.abspath(out_parquet):
                raise OSError("Simulated filesystem locked during final replacement")
            return real_move(src, dst, **kwargs)

        with mock.patch("shutil.move", side_effect=failing_move):
            with pytest.raises(OSError, match="Simulated filesystem locked during final replacement"):
                pipeline.write_parquet_bounded_arrow(features_df, out_parquet)

        # Verification 2: Rollback restored the original valid output from backup!
        assert os.path.exists(out_parquet), "Output directory missing after failed replacement!"
        assert os.path.exists(sentinel_path), "Sentinel missing: rollback failed to restore original directory!"
        with open(sentinel_path, "r", encoding="utf-8") as f:
            assert f.read() == "VALID_DATA_V1"
        # No orphan backup or staging directories
        orphan_dirs = [d for d in os.listdir(test_scratch_dir) if d.startswith(".staging_") or d.startswith(".backup_")]
        assert len(orphan_dirs) == 0, f"Orphan dirs after rollback: {orphan_dirs}"

    finally:
        spark.stop()
        if os.path.exists(test_scratch_dir):
            shutil.rmtree(test_scratch_dir, ignore_errors=True)




