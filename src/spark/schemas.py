"""
Explicit schemas and data contracts for Apache Spark Zenodo Bitcoin transaction processing.

SCHEMA COLUMN TAXONOMY & COUNT SPECIFICATION:
=============================================
1. Physical Parquet Columns (18 columns):
   The columns physically serialized into binary Parquet files (part-0.parquet) via get_transaction_features_pyarrow_schema():
   - Identifiers (3): block_number, tx_index, tx_hash
   - Numerical Model Features (15): num_inputs, num_outputs, total_input_satoshi, total_output_satoshi,
     fee_satoshi, fee_rate, is_coinbase, avg_input_val, avg_output_val, input_output_count_ratio,
     input_output_val_ratio, max_input_prior_tx_count, avg_input_prior_tx_count,
     max_input_prior_volume_satoshi, new_address_fraction.

2. Hive Partition Directory Key (1 column):
   - block_chunk (e.g. 'block_chunk=480000-480999'):
     Encoded strictly in the filesystem directory hierarchy to enable chunk-pruning partition discovery.
     Not duplicated inside the physical row payload of each Parquet file.

3. Partitioned Dataset Columns (19 columns):
   When loaded via pyarrow.dataset.dataset(..., partitioning='hive') or spark.read.parquet(...),
   the 18 physical Parquet columns plus the synthesized partition column 'block_chunk' yield
   exactly 19 dataset columns.

4. Machine Learning Model Features (15 features):
   The 15 numerical and structural features consumed by anomaly detection models (excluding
   the 4 identifiers/metadata keys: block_number, tx_index, tx_hash, block_chunk).
"""
from typing import Dict, Any, List

# Explicit feature list used by downstream machine learning models
MODEL_FEATURE_COLUMNS: List[str] = [
    "num_inputs",
    "num_outputs",
    "total_input_satoshi",
    "total_output_satoshi",
    "fee_satoshi",
    "fee_rate",
    "is_coinbase",
    "avg_input_val",
    "avg_output_val",
    "input_output_count_ratio",
    "input_output_val_ratio",
    "max_input_prior_tx_count",
    "avg_input_prior_tx_count",
    "max_input_prior_volume_satoshi",
    "new_address_fraction"
]

METADATA_IDENTIFIER_COLUMNS: List[str] = [
    "block_number",
    "tx_index",
    "tx_hash",
    "block_chunk"
]

# ------------------------------------------------------------------------------
# 1. Platform-Agnostic Schema Specifications
# ------------------------------------------------------------------------------

ZENODO_RAW_RECORD_SPEC = {
    "kind": "string",          # "BTC-IN" or "BTC-OUT"
    "block_number": "long",    # Block height (e.g. 480000)
    "tx_index": "integer",     # Intra-block transaction index (0 for coinbase)
    "source": "string",        # Input address or coinbase marker "0"
    "destination": "string",   # Output address or 64-hex transaction ID
    "value_satoshi": "long"    # Satoshi transfer value
}

COMPLETED_TRANSACTION_SPEC = {
    "block_number": "long",
    "tx_index": "integer",
    "tx_hash": "string",
    "num_inputs": "integer",
    "num_outputs": "integer",
    "total_input_satoshi": "long",
    "total_output_satoshi": "long",
    "fee_satoshi": "long",
    "is_coinbase": "boolean",
    "block_chunk": "string"    # Partition key, e.g. "480000-480999"
}

TRANSACTION_FEATURE_SPEC = {
    "block_number": "long",
    "tx_index": "integer",
    "tx_hash": "string",
    "num_inputs": "integer",
    "num_outputs": "integer",
    "total_input_satoshi": "long",
    "total_output_satoshi": "long",
    "fee_satoshi": "long",
    "fee_rate": "double",
    "is_coinbase": "integer",
    "avg_input_val": "double",
    "avg_output_val": "double",
    "input_output_count_ratio": "double",
    "input_output_val_ratio": "double",
    "max_input_prior_tx_count": "long",
    "avg_input_prior_tx_count": "double",
    "max_input_prior_volume_satoshi": "long",
    "new_address_fraction": "double",
    "block_chunk": "string"
}


# ------------------------------------------------------------------------------
# 2. PySpark StructType Schemas (lazily initialized if pyspark is available)
# ------------------------------------------------------------------------------

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        LongType,
        IntegerType,
        DoubleType,
        BooleanType,
        ArrayType
    )
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False


def get_raw_record_spark_schema():
    """Return PySpark StructType schema for raw Zenodo records."""
    if not HAS_PYSPARK:
        raise ImportError(
            "PySpark is not installed in the current environment. "
            "Install pyspark (e.g., 'pip install pyspark==3.5.0') to instantiate Spark StructType schemas."
        )
    return StructType([
        StructField("kind", StringType(), False),
        StructField("block_number", LongType(), False),
        StructField("tx_index", IntegerType(), False),
        StructField("source", StringType(), False),
        StructField("destination", StringType(), False),
        StructField("value_satoshi", LongType(), False),
    ])


def get_completed_transaction_spark_schema():
    """Return PySpark StructType schema for assembled completed transactions."""
    if not HAS_PYSPARK:
        raise ImportError(
            "PySpark is not installed in the current environment. "
            "Install pyspark (e.g., 'pip install pyspark==3.5.0') to instantiate Spark StructType schemas."
        )
    return StructType([
        StructField("block_number", LongType(), False),
        StructField("tx_index", IntegerType(), False),
        StructField("tx_hash", StringType(), False),
        StructField("num_inputs", IntegerType(), False),
        StructField("num_outputs", IntegerType(), False),
        StructField("total_input_satoshi", LongType(), False),
        StructField("total_output_satoshi", LongType(), False),
        StructField("fee_satoshi", LongType(), True),
        StructField("is_coinbase", BooleanType(), False),
        StructField("block_chunk", StringType(), False),
    ])


def get_transaction_features_spark_schema():
    """Return PySpark StructType schema for derived historical transaction features."""
    if not HAS_PYSPARK:
        raise ImportError(
            "PySpark is not installed in the current environment. "
            "Install pyspark (e.g., 'pip install pyspark==3.5.0') to instantiate Spark StructType schemas."
        )
    return StructType([
        StructField("block_number", LongType(), False),
        StructField("tx_index", IntegerType(), False),
        StructField("tx_hash", StringType(), False),
        StructField("num_inputs", IntegerType(), False),
        StructField("num_outputs", IntegerType(), False),
        StructField("total_input_satoshi", LongType(), False),
        StructField("total_output_satoshi", LongType(), False),
        StructField("fee_satoshi", LongType(), True),
        StructField("fee_rate", DoubleType(), True),
        StructField("is_coinbase", IntegerType(), False),
        StructField("avg_input_val", DoubleType(), True),
        StructField("avg_output_val", DoubleType(), True),
        StructField("input_output_count_ratio", DoubleType(), True),
        StructField("input_output_val_ratio", DoubleType(), True),
        StructField("max_input_prior_tx_count", LongType(), True),
        StructField("avg_input_prior_tx_count", DoubleType(), True),
        StructField("max_input_prior_volume_satoshi", LongType(), True),
        StructField("new_address_fraction", DoubleType(), True),
        StructField("block_chunk", StringType(), False),
    ])


def get_transaction_features_pyarrow_schema():
    """Return PyArrow schema corresponding to TRANSACTION_FEATURE_SPEC (excluding partition key)."""
    import pyarrow as pa
    return pa.schema([
        ("block_number", pa.int64()),
        ("tx_index", pa.int32()),
        ("tx_hash", pa.string()),
        ("num_inputs", pa.int32()),
        ("num_outputs", pa.int32()),
        ("total_input_satoshi", pa.int64()),
        ("total_output_satoshi", pa.int64()),
        ("fee_satoshi", pa.int64()),
        ("fee_rate", pa.float64()),
        ("is_coinbase", pa.int32()),
        ("avg_input_val", pa.float64()),
        ("avg_output_val", pa.float64()),
        ("input_output_count_ratio", pa.float64()),
        ("input_output_val_ratio", pa.float64()),
        ("max_input_prior_tx_count", pa.int64()),
        ("avg_input_prior_tx_count", pa.float64()),
        ("max_input_prior_volume_satoshi", pa.int64()),
        ("new_address_fraction", pa.float64()),
    ])

