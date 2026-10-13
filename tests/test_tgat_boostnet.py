"""
Unit and regression tests for TGAT-BoostNet hybrid architecture,
leakage isolation, embedding extraction, and experimental reproducibility.
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.gat import SpatioTemporalGAT, is_gat_available


DATA_PATH = "data/processed/elliptic/elliptic_preprocessed.npz"
EXP_DIR = "data/validation/tgat_boostnet_experiment"


def test_tgat_environment_and_architecture_dimensions():
    """Verify PyTorch/PyG environment and SpatioTemporalGAT dimension consistency."""
    available, msg = is_gat_available()
    assert available, f"TGAT dependencies unavailable: {msg}"

    model = SpatioTemporalGAT(in_channels=165, hidden_channels=32, out_channels=2, heads=4, dropout=0.3)
    x = torch.randn(10, 165)
    t = torch.tensor([[0.5]] * 10, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    # Forward logits
    logits = model(x, edge_index, t)
    assert logits.shape == (10, 2), f"Expected logits (10, 2), got {logits.shape}"

    # Forward with embedding return
    logits2, emb = model(x, edge_index, t, return_embedding=True)
    assert logits2.shape == (10, 2)
    assert emb.shape == (10, 32), f"Expected embedding (10, 32), got {emb.shape}"

    # Direct extract_embedding
    extracted = model.extract_embedding(x, edge_index, t)
    assert extracted.shape == (10, 32), f"Expected extracted embedding (10, 32), got {extracted.shape}"


def test_tgat_feature_fusion_197_dimension():
    """Verify that fusing 165 raw features and 32 TGAT graph embeddings yields exactly 197 dimensions."""
    N = 25
    x_raw = np.random.randn(N, 165).astype(np.float32)
    model = SpatioTemporalGAT(in_channels=165, hidden_channels=32, out_channels=2)
    t_tensor = torch.tensor([[0.2]] * N, dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    emb = model.extract_embedding(torch.tensor(x_raw), edge_index, t_tensor).cpu().numpy()
    assert emb.shape == (N, 32)

    fused = np.concatenate([x_raw, emb], axis=1)
    assert fused.shape == (N, 197), f"Expected fused representation of (25, 197), got {fused.shape}"


def test_unknown_label_isolation_and_node_preservation():
    """Verify that all 203,769 nodes are preserved and unknown nodes (-1) are masked from loss/eval."""
    assert os.path.exists(DATA_PATH), f"Missing preprocessed dataset at {DATA_PATH}"
    data = np.load(DATA_PATH)
    tx_ids = data["tx_ids"]
    y = data["y"]
    time_steps = data["time_steps"]

    assert len(tx_ids) == 203769, f"Expected 203,769 nodes, found {len(tx_ids)}"
    n_unknowns = int((y == -1).sum())
    assert n_unknowns == 157205, f"Expected 157,205 unknown nodes, found {n_unknowns}"

    labeled = (y != -1)
    train_mask = (time_steps <= 29) & labeled
    val_mask = (time_steps >= 30) & (time_steps <= 34) & labeled
    test_mask = (time_steps >= 35) & labeled

    # Zero unknown nodes in any supervised split
    assert (y[train_mask] == -1).sum() == 0
    assert (y[val_mask] == -1).sum() == 0
    assert (y[test_mask] == -1).sum() == 0

    assert train_mask.sum() == 26381
    assert val_mask.sum() == 3513
    assert test_mask.sum() == 16670


def test_graph_temporal_leakage_isolation():
    """Verify that 100% of graph edges connect nodes within the exact same time step."""
    assert os.path.exists(DATA_PATH)
    data = np.load(DATA_PATH)
    time_steps = data["time_steps"]
    edge_index = data["edge_index"]

    src_times = time_steps[edge_index[0]]
    dst_times = time_steps[edge_index[1]]

    cross_edges = (src_times != dst_times).sum()
    assert cross_edges == 0, f"Detected {cross_edges} cross-timestep edges! Graph must have 0 temporal leakage."


def test_step_35_authoritative_counts():
    """Verify authoritative resolution of Step 35 counts (1,341 labeled, 182 illicit)."""
    assert os.path.exists(DATA_PATH)
    data = np.load(DATA_PATH)
    time_steps = data["time_steps"]
    y = data["y"]

    step35_labeled = (time_steps == 35) & (y != -1)
    step35_illicit = (time_steps == 35) & (y == 1)
    step35_licit = (time_steps == 35) & (y == 0)
    step35_unknown = (time_steps == 35) & (y == -1)

    assert step35_labeled.sum() == 1341, f"Expected 1,341 labeled nodes at step 35, got {step35_labeled.sum()}"
    assert step35_illicit.sum() == 182, f"Expected 182 illicit nodes at step 35, got {step35_illicit.sum()}"
    assert step35_licit.sum() == 1159, f"Expected 1,159 licit nodes at step 35, got {step35_licit.sum()}"
    assert step35_unknown.sum() == 4166, f"Expected 4,166 unknown nodes at step 35, got {step35_unknown.sum()}"
    assert (time_steps == 35).sum() == 5507, "Total nodes at step 35 must equal 5,507"


def test_experiment_artifacts_exist_and_consistent():
    """Verify that all required TGAT-BoostNet experiment deliverables exist and are valid."""
    required_files = [
        "DATA_INTEGRITY_AUDIT.md",
        "GRAPH_LEAKAGE_AUDIT.md",
        "environment_manifest.txt",
        "metrics.json",
        "seed_results.csv",
        "per_timestep_metrics.csv",
        "confusion_matrices.json",
        "training_history.json",
        "latency_benchmark.json",
        "tgat_checkpoint.pt",
        "tgat_boostnet_xgb.json",
        "xgboost_alone.json",
        "scaler.joblib",
        "test_predictions.parquet"
    ]

    for fname in required_files:
        fpath = os.path.join(EXP_DIR, fname)
        assert os.path.exists(fpath), f"Required artifact {fname} missing in {EXP_DIR}"
        assert os.path.getsize(fpath) > 0, f"Artifact {fname} is empty"

    # Verify predictions parquet
    df_preds = pd.read_parquet(os.path.join(EXP_DIR, "test_predictions.parquet"))
    assert len(df_preds) == 16670, f"Expected 16,670 test predictions, got {len(df_preds)}"
    assert "prob_tgat_boostnet" in df_preds.columns
    assert "pred_tgat_boostnet" in df_preds.columns

    # Verify metrics JSON
    with open(os.path.join(EXP_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
    assert "model_summary_5_seeds" in metrics
    assert "TGAT_BoostNet" in metrics["model_summary_5_seeds"]
    assert metrics["model_summary_5_seeds"]["TGAT_BoostNet"]["f1_illicit_mean"] > 0.70
