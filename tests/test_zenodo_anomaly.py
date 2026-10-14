"""
Zenodo anomaly pipeline tests: Spark/pandas feature parity, data-quality rules, deterministic
sampling, persisted-output integrity (when the real outputs exist).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.anomaly.zenodo_features import (INPUT_COLUMNS, MODEL_FEATURES, compute_features, dq_status,  # noqa: E402
                                         sample_bucket)

FEAT_DIR = os.path.join(ROOT, "data/processed/zenodo_anomaly/features")
SCORE_DIR = os.path.join(ROOT, "data/processed/zenodo_anomaly/scores")
SRC_DIR = os.path.join(ROOT, "data/parquet/zenodo_features")


def rows():
    return pd.DataFrame({
        "block_number": [485000, 485000, 485001, 485001], "tx_index": [0, 1, 2, 3],
        "tx_hash": ["a" * 64, "b" * 64, "c" * 64, "NOT-HEX"],
        "num_inputs": [1, 2, 1, 1], "num_outputs": [1, 3, 1, 1],
        "total_input_satoshi": [625_000_000, 1_000_000, 50_000, 10], "total_output_satoshi": [625_000_000, 990_000, 50_000, 10],
        "fee_satoshi": [0, 10_000, 5, 0], "fee_rate": [0.0, 2000.0, 2.5, 0.0], "is_coinbase": [1, 0, 0, 0],
        "avg_input_val": [625e6, 5e5, 5e4, 10.0], "avg_output_val": [625e6, 330000.0, 5e4, 10.0],
        "input_output_count_ratio": [1.0, 2 / 3, 1.0, 1.0], "input_output_val_ratio": [1.0, 1.0101, 1.0, 1.0],
        "max_input_prior_tx_count": [0, 4, 0, 0], "avg_input_prior_tx_count": [0.0, 2.5, 0.0, 0.0],
        "max_input_prior_volume_satoshi": [0, 7_000_000, 0, 0], "new_address_fraction": [0.0, 0.5, 1.0, 1.0]})


def test_dq_rules():
    st = dq_status(rows())
    assert list(st) == ["ok", "ok", "invalid:fee_mismatch", "invalid:bad_tx_hash"]


def test_features_finite_and_ordered():
    f = compute_features(rows())
    assert list(f.columns) == MODEL_FEATURES
    assert np.isfinite(f.to_numpy()).all()
    assert f.loc[1, "log_num_inputs"] == pytest.approx(np.log1p(2))
    assert f.loc[1, "log_io_count_ratio"] == pytest.approx(np.log(2 / 3))


def test_sample_bucket_deterministic():
    h = pd.Series(["0000000a" + "0" * 56, "ffffffff" + "0" * 56])
    assert list(sample_bucket(h)) == [10, 0xFFFFFFFF % 10000]
    assert np.array_equal(sample_bucket(h), sample_bucket(h.copy()))


@pytest.mark.skipif(not os.path.isdir(FEAT_DIR), reason="Zenodo feature stage not run")
def test_spark_features_match_pandas_definition():
    dset = ds.dataset(FEAT_DIR, format="parquet", partitioning="hive")
    t = dset.to_table(columns=INPUT_COLUMNS + MODEL_FEATURES + ["sample_bucket", "dq_status"],
                      filter=(ds.field("block_chunk") == "487000-487999") & (ds.field("sample_bucket") < 5)).to_pandas()
    assert len(t) > 500
    ref = compute_features(t[INPUT_COLUMNS])
    np.testing.assert_allclose(t[MODEL_FEATURES].to_numpy(), ref.to_numpy(), rtol=1e-9, atol=1e-9)
    assert np.array_equal(t["sample_bucket"].to_numpy(), sample_bucket(t["tx_hash"]))
    assert (t["dq_status"].to_numpy() == dq_status(t[INPUT_COLUMNS])).all()


@pytest.mark.skipif(not os.path.isdir(SCORE_DIR), reason="Zenodo scoring stage not run")
def test_persisted_scores_cover_source_rows():
    src = ds.dataset(SRC_DIR, format="parquet", partitioning="hive")
    sc = ds.dataset(SCORE_DIR, format="parquet", partitioning="hive")
    assert sc.count_rows() == src.count_rows()
    summary = json.load(open(os.path.join(ROOT, "data/processed/zenodo_anomaly/score_summary.json")))
    assert summary["total_rows"] == src.count_rows()
    chunk = sc.to_table(columns=["score_status", "score", "percentile", "is_coinbase"],
                        filter=ds.field("block_chunk") == "489000-489999").to_pandas()
    scored = chunk[chunk.score_status == "scored"]
    assert scored["score"].notna().all() and scored["percentile"].between(0, 1).all()
    assert chunk.loc[chunk.score_status != "scored", "score"].isna().all()
    assert (chunk.loc[chunk.is_coinbase == 1, "score_status"] == "skipped_coinbase").all()
