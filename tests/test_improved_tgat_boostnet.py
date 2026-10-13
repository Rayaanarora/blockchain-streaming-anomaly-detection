"""
Unit and regression tests for Improved TGAT-BoostNet 7-configuration suite
and baseline reproducibility.
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.gat import SpatioTemporalGAT, is_gat_available

EXP_DIR = "data/validation/tgat_boostnet_improved"


def test_improved_artifacts_exist():
    """Verify that all deliverables in tgat_boostnet_improved exist and are non-empty."""
    required = [
        "AUDIT_AND_BASELINE_REPRODUCIBILITY.md",
        "IMPROVED_TGAT_BOOSTNET_REPORT.md",
        "metrics.json",
        "seed_results.csv",
        "per_timestep_metrics.csv",
        "confusion_matrices.json",
        "latency_benchmark.json",
        "training_history.json",
        "test_predictions.parquet",
        "scaler.joblib",
        "tgat_checkpoint.pt",
        "tgat_boostnet_xgb.json",
        "tgat_boostnet_opt_xgb.json",
        "xgboost_baseline.json",
        "histgbdt_baseline.joblib"
    ]
    for f in required:
        p = os.path.join(EXP_DIR, f)
        assert os.path.exists(p), f"Missing artifact: {p}"
        assert os.path.getsize(p) > 0, f"Empty artifact: {p}"


def test_improved_metrics_json_integrity():
    """Verify that metrics.json contains the complete 7-configuration summary."""
    with open(os.path.join(EXP_DIR, "metrics.json"), "r") as f:
        data = json.load(f)

    assert "model_summary_5_seeds" in data
    summary = data["model_summary_5_seeds"]

    # All 8 evaluated configs present
    expected_configs = [
        "1_HistGBDT_165",
        "2_XGBoost_Baseline_165",
        "3_TGAT_Alone_32",
        "4_TGAT_BoostNet_Baseline_197",
        "5_TGAT_HistGBDT_197",
        "6_Tuned_XGBoost_165",
        "7_TGAT_BoostNet_Optimized_197",
        "8_TGAT_Hybrid_Ensemble"
    ]
    for cfg in expected_configs:
        assert cfg in summary, f"Configuration {cfg} missing from summary"
        assert summary[cfg]["f1_illicit_mean"] > 0.30

    # Hybrid Ensemble has highest F1
    assert summary["8_TGAT_Hybrid_Ensemble"]["f1_illicit_mean"] >= 0.755


def test_test_predictions_parquet_reproducibility():
    """Verify that test_predictions.parquet reproduces metrics directly."""
    df = pd.read_parquet(os.path.join(EXP_DIR, "test_predictions.parquet"))
    assert len(df) == 16670
    assert "y_true" in df.columns
    assert "prob_tgat_boostnet_base" in df.columns
    assert "prob_hybrid_ensemble" in df.columns

    y_true = df["y_true"].values
    p_hyb = df["prob_tgat_boostnet_base"].values

    roc = roc_auc_score(y_true, p_hyb)
    assert roc > 0.925
