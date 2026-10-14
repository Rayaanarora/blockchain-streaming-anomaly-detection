"""
Zenodo anomaly-detection feature contract (version 1).

Inputs are the 18 physical columns already produced by the Spark Zenodo batch pipeline
(data/parquet/zenodo_features). The Parquet store does NOT contain addresses, per-output
values, block timestamps or transaction graph edges, so only transaction-level and
address-history aggregate features are available. "Graph" information enters only through
the historical address-participation features computed during preprocessing
(max/avg_input_prior_tx_count, max_input_prior_volume_satoshi, new_address_fraction).

Every model feature is a monotone, heavy-tail-compressing transform of a stored column
(log1p of counts and satoshi amounts), so robust statistics and Isolation Forest splits are
not dominated by a few huge values. The same definitions are implemented twice - as Spark
column expressions (batch) and as a pandas function (streaming / API) - and
tests/test_zenodo_anomaly.py checks they agree.
"""
import numpy as np
import pandas as pd

FEATURE_VERSION = "zenodo-features-v1"

INPUT_COLUMNS = [
    "block_number", "tx_index", "tx_hash", "num_inputs", "num_outputs", "total_input_satoshi",
    "total_output_satoshi", "fee_satoshi", "fee_rate", "is_coinbase", "avg_input_val", "avg_output_val",
    "input_output_count_ratio", "input_output_val_ratio", "max_input_prior_tx_count", "avg_input_prior_tx_count",
    "max_input_prior_volume_satoshi", "new_address_fraction",
]

MODEL_FEATURES = [
    "log_num_inputs",            # log1p(num_inputs)
    "log_num_outputs",           # log1p(num_outputs)
    "log_total_output",          # log1p(total_output_satoshi)
    "log_fee",                   # log1p(fee_satoshi)
    "log_fee_per_io",            # log1p(fee_rate); fee_rate = fee / (num_inputs + num_outputs), NOT per byte
    "log_fee_bps_of_input",      # log1p(10000 * fee / total_input)
    "log_avg_input_val",         # log1p(avg_input_val)
    "log_avg_output_val",        # log1p(avg_output_val)
    "log_io_count_ratio",        # log(num_inputs / num_outputs)
    "log_max_input_prior_tx",    # log1p(max_input_prior_tx_count)
    "log_avg_input_prior_tx",    # log1p(avg_input_prior_tx_count)
    "log_max_input_prior_volume",  # log1p(max_input_prior_volume_satoshi)
    "new_addr_fraction",         # = new_address_fraction; fraction of input addresses never seen before in the processed window
]

# Data-quality rule ids (row-level). A row failing any rule is not used for fitting and is
# scored with data_quality_status = "invalid:<rule>".
DQ_RULES = ["null_key", "bad_tx_hash", "no_outputs", "no_inputs_non_coinbase", "negative_fee",
            "fee_mismatch", "negative_value"]


def spark_dq_status(F):
    """Spark column expression returning 'ok' or 'invalid:<first failing rule>'."""
    c = F.col
    return (F.when(c("block_number").isNull() | c("tx_index").isNull() | c("tx_hash").isNull(), "invalid:null_key")
            .when(~c("tx_hash").rlike("^[0-9a-f]{64}$"), "invalid:bad_tx_hash")
            .when(c("num_outputs") < 1, "invalid:no_outputs")
            .when((c("is_coinbase") == 0) & (c("num_inputs") < 1), "invalid:no_inputs_non_coinbase")
            .when(c("fee_satoshi") < 0, "invalid:negative_fee")
            .when((c("is_coinbase") == 0) &
                  (c("fee_satoshi") != c("total_input_satoshi") - c("total_output_satoshi")), "invalid:fee_mismatch")
            .when((c("total_input_satoshi") < 0) | (c("total_output_satoshi") < 0), "invalid:negative_value")
            .otherwise("ok"))


def spark_feature_columns(F):
    """List of Spark column expressions aliased to MODEL_FEATURES."""
    c = F.col
    ln = lambda e: F.log1p(F.greatest(e.cast("double"), F.lit(0.0)))  # noqa: E731
    return [
        ln(c("num_inputs")).alias("log_num_inputs"),
        ln(c("num_outputs")).alias("log_num_outputs"),
        ln(c("total_output_satoshi")).alias("log_total_output"),
        ln(c("fee_satoshi")).alias("log_fee"),
        ln(c("fee_rate")).alias("log_fee_per_io"),
        ln(F.when(c("total_input_satoshi") > 0, F.lit(10000.0) * c("fee_satoshi") / c("total_input_satoshi"))
           .otherwise(0.0)).alias("log_fee_bps_of_input"),
        ln(c("avg_input_val")).alias("log_avg_input_val"),
        ln(c("avg_output_val")).alias("log_avg_output_val"),
        F.when((c("num_inputs") > 0) & (c("num_outputs") > 0),
               F.log(c("num_inputs").cast("double") / c("num_outputs").cast("double"))).otherwise(0.0)
        .alias("log_io_count_ratio"),
        ln(c("max_input_prior_tx_count")).alias("log_max_input_prior_tx"),
        ln(c("avg_input_prior_tx_count")).alias("log_avg_input_prior_tx"),
        ln(c("max_input_prior_volume_satoshi")).alias("log_max_input_prior_volume"),
        F.coalesce(c("new_address_fraction").cast("double"), F.lit(0.0)).alias("new_addr_fraction"),
    ]


def sample_bucket_spark(F):
    """Deterministic 0..9999 bucket from the first 8 hex chars of tx_hash (order-independent sampling)."""
    return (F.conv(F.substring(F.col("tx_hash"), 1, 8), 16, 10).cast("long") % 10000).cast("int")


def sample_bucket(tx_hash: pd.Series) -> np.ndarray:
    return (tx_hash.str.slice(0, 8).apply(lambda h: int(h, 16)).to_numpy(np.int64) % 10000).astype(np.int32)


def dq_status(df: pd.DataFrame) -> np.ndarray:
    status = np.array(["ok"] * len(df), dtype=object)
    nonzero = lambda m: m & (status == "ok")  # noqa: E731 - first failing rule wins
    rules = [
        ("null_key", df[["block_number", "tx_index", "tx_hash"]].isna().any(axis=1).to_numpy()),
        ("bad_tx_hash", ~df["tx_hash"].fillna("").str.fullmatch(r"[0-9a-f]{64}").to_numpy()),
        ("no_outputs", (df["num_outputs"] < 1).to_numpy()),
        ("no_inputs_non_coinbase", ((df["is_coinbase"] == 0) & (df["num_inputs"] < 1)).to_numpy()),
        ("negative_fee", (df["fee_satoshi"] < 0).to_numpy()),
        ("fee_mismatch", ((df["is_coinbase"] == 0) &
                          (df["fee_satoshi"] != df["total_input_satoshi"] - df["total_output_satoshi"])).to_numpy()),
        ("negative_value", ((df["total_input_satoshi"] < 0) | (df["total_output_satoshi"] < 0)).to_numpy()),
    ]
    for name, mask in rules:
        status[nonzero(mask)] = f"invalid:{name}"
    return status


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """pandas implementation of spark_feature_columns (same formulas)."""
    ln = lambda s: np.log1p(np.maximum(s.astype("float64").to_numpy(), 0.0))  # noqa: E731
    tin = df["total_input_satoshi"].astype("float64").to_numpy()
    fee = df["fee_satoshi"].astype("float64").to_numpy()
    nin = df["num_inputs"].astype("float64").to_numpy()
    nout = df["num_outputs"].astype("float64").to_numpy()
    bps = np.where(tin > 0, 10000.0 * fee / np.where(tin > 0, tin, 1.0), 0.0)
    ratio = np.where((nin > 0) & (nout > 0), np.log(np.where(nin > 0, nin, 1.0) / np.where(nout > 0, nout, 1.0)), 0.0)
    out = pd.DataFrame({
        "log_num_inputs": ln(df["num_inputs"]),
        "log_num_outputs": ln(df["num_outputs"]),
        "log_total_output": ln(df["total_output_satoshi"]),
        "log_fee": ln(df["fee_satoshi"]),
        "log_fee_per_io": ln(df["fee_rate"]),
        "log_fee_bps_of_input": np.log1p(np.maximum(bps, 0.0)),
        "log_avg_input_val": ln(df["avg_input_val"]),
        "log_avg_output_val": ln(df["avg_output_val"]),
        "log_io_count_ratio": ratio,
        "log_max_input_prior_tx": ln(df["max_input_prior_tx_count"]),
        "log_avg_input_prior_tx": ln(df["avg_input_prior_tx_count"]),
        "log_max_input_prior_volume": ln(df["max_input_prior_volume_satoshi"]),
        "new_addr_fraction": df["new_address_fraction"].astype("float64").fillna(0.0).to_numpy(),
    }, index=df.index)
    return out[MODEL_FEATURES]
