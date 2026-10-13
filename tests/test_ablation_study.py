"""
Unit and regression tests for the comprehensive model ablation study,
feature audit, stochastic seeds, imbalance strategies, and temporal drift.
"""
import os
import json
import pytest
import numpy as np
import pandas as pd


ABLATION_DIR = os.path.join("data", "validation", "ablation_study")


def test_feature_taxonomy_resolution():
    """Verify exact count and boundaries of local vs aggregated features."""
    audit_file = os.path.join(ABLATION_DIR, "feature_audit.json")
    if not os.path.exists(audit_file):
        pytest.skip(f"Audit file not found at {audit_file}")

    with open(audit_file, "r") as f:
        data = json.load(f)

    tax = data["taxonomy_resolution"]
    assert tax["total_feature_count"] == 165
    assert tax["local_feature_count"] == 93
    assert tax["aggregated_feature_count"] == 72
    assert tax["local_feature_count"] + tax["aggregated_feature_count"] == 165
    assert tax["local_feature_range_1_indexed"] == "f1 to f93"
    assert tax["aggregated_feature_range_1_indexed"] == "f94 to f165"


def test_ablation_artifacts_exist():
    """Verify all 7 required ablation artifacts exist and are non-empty."""
    expected_files = [
        "ablation_metrics.json",
        "feature_audit.json",
        "imbalance_ablation.csv",
        "per_step_metrics.csv",
        "seed_ablation.csv",
        "per_step_f1_drift.svg",
        "prevalence_and_samples.svg"
    ]
    for fname in expected_files:
        fpath = os.path.join(ABLATION_DIR, fname)
        assert os.path.exists(fpath), f"Missing artifact: {fpath}"
        assert os.path.getsize(fpath) > 100, f"Artifact is too small/empty: {fpath}"


def test_seed_ablation_reproducibility():
    """Verify 5 seeds evaluated and deterministic vs stochastic properties."""
    seed_file = os.path.join(ABLATION_DIR, "seed_ablation.csv")
    if not os.path.exists(seed_file):
        pytest.skip(f"Seed ablation file not found at {seed_file}")

    df_seeds = pd.read_csv(seed_file)
    assert len(df_seeds) == 15  # 3 models x 5 seeds

    for model_name in ["gbdt", "rf", "lr"]:
        sub = df_seeds[df_seeds["model"] == model_name]
        assert len(sub) == 5
        seeds = sorted(sub["seed"].tolist())
        assert seeds == [42, 123, 456, 789, 2024]

        # All runs must have positive ROC-AUC and F1
        assert (sub["test_roc_auc"] >= 0.80).all()
        assert (sub["test_f1"] >= 0.20).all()

    # LR must be completely deterministic (zero variance across seeds)
    lr_f1 = df_seeds[df_seeds["model"] == "lr"]["test_f1"].values
    assert np.std(lr_f1) == 0.0

    # HistGBDT and RF must exhibit non-zero stochastic seed variance
    gbdt_f1 = df_seeds[df_seeds["model"] == "gbdt"]["test_f1"].values
    rf_f1 = df_seeds[df_seeds["model"] == "rf"]["test_f1"].values
    assert np.std(gbdt_f1) > 0.0
    assert np.std(rf_f1) > 0.0


def test_imbalance_strategies_contract():
    """Verify training-set class weighting vs random undersampling comparisons."""
    imb_file = os.path.join(ABLATION_DIR, "imbalance_ablation.csv")
    if not os.path.exists(imb_file):
        pytest.skip(f"Imbalance ablation file not found at {imb_file}")

    df_imb = pd.read_csv(imb_file)
    assert len(df_imb) == 12  # 4 strategies x 3 models

    strategies = df_imb["strategy"].unique()
    assert len(strategies) == 4

    # Undersampling 1:1 training samples must equal 5,742 (2,871 licit + 2,871 illicit)
    rus_1to1 = df_imb[df_imb["strategy"].str.contains("1:1")]
    assert (rus_1to1["training_samples"] == 5742).all()

    # Undersampling 2:1 training samples must equal 8,613 (5,742 licit + 2,871 illicit)
    rus_2to1 = df_imb[df_imb["strategy"].str.contains("2:1")]
    assert (rus_2to1["training_samples"] == 8613).all()

    # Full training sets must have 26,381 samples
    cw = df_imb[df_imb["strategy"].str.contains("Cost-Sensitive")]
    assert (cw["training_samples"] == 26381).all()


def test_temporal_drift_instability_flags():
    """Verify small-sample instability flags for steps 45 & 46 and step 35 performance."""
    temp_file = os.path.join(ABLATION_DIR, "per_step_metrics.csv")
    if not os.path.exists(temp_file):
        pytest.skip(f"Temporal metrics file not found at {temp_file}")

    df_temp = pd.read_csv(temp_file)
    assert len(df_temp) == 15  # steps 35 to 49

    # Step 35 immediate post-validation step must achieve F1 > 0.90 for GBDT and RF
    s35 = df_temp[df_temp["time_step"] == 35].iloc[0]
    assert s35["gbdt_f1"] > 0.90
    assert s35["rf_f1"] > 0.90

    # Steps 45 & 46 have illicit count <= 5, must be flagged HIGH_ESTIMATOR_VARIANCE
    s45 = df_temp[df_temp["time_step"] == 45].iloc[0]
    s46 = df_temp[df_temp["time_step"] == 46].iloc[0]
    assert s45["illicit_count"] == 5
    assert s46["illicit_count"] == 2
    assert "HIGH_ESTIMATOR_VARIANCE" in s45["stability_status"]
    assert "HIGH_ESTIMATOR_VARIANCE" in s46["stability_status"]


def test_gat_message_passing_temporal_integrity():
    """Verify zero cross-timestep edges exist in real Elliptic graph."""
    data_path = os.path.join("data", "processed", "elliptic", "elliptic_preprocessed.npz")
    if not os.path.exists(data_path):
        pytest.skip("Processed dataset not found")

    d = np.load(data_path)
    times = d["time_steps"]
    edges = d["edge_index"]

    src_times = times[edges[0]]
    dst_times = times[edges[1]]
    diff = dst_times - src_times

    # Strict: All edges must connect nodes within the exact same time step
    assert (diff == 0).all(), f"Found {(diff != 0).sum()} cross-timestep edges!"
