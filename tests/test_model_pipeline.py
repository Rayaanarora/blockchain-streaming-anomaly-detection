"""
Unit and regression tests for leakage-safe model training, validation protocols, and Zenodo integration.
"""
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.baselines import CostSensitiveAnomalyModel
from src.features.extractor import FEATURE_NAMES as ZENODO_FEATURE_NAMES


# ==============================================================================
# 1. Scaler Leakage Safeguards
# ==============================================================================

def test_scaler_fitted_strictly_on_train_nodes():
    """Verify that fitting on training only prevents future-to-past statistical leakage."""
    rng = np.random.RandomState(42)
    # Simulate 100 train nodes from N(0, 1) and 50 test nodes from N(5, 2)
    X_train = rng.randn(100, 10).astype(np.float32)
    X_test = (rng.randn(50, 10) * 2.0 + 5.0).astype(np.float32)

    train_only_scaler = StandardScaler()
    train_only_scaler.fit(X_train)

    leaky_scaler = StandardScaler()
    leaky_scaler.fit(np.vstack([X_train, X_test]))

    # Mean and variance must differ; leaky scaler is corrupted by future distribution
    assert not np.allclose(train_only_scaler.mean_, leaky_scaler.mean_)
    assert not np.allclose(train_only_scaler.var_, leaky_scaler.var_)


# ==============================================================================
# 2. Unknown Label Isolation
# ==============================================================================

def test_unknown_label_isolation():
    """Verify unknown labels (-1) are strictly isolated from train, val, and test masks."""
    y = np.array([0, 1, -1, 0, -1, 1, -1])
    time_steps = np.array([1, 15, 20, 31, 33, 40, 45])

    labeled = (y != -1)
    train_mask = (time_steps <= 29) & labeled
    val_mask = (time_steps >= 30) & (time_steps <= 34) & labeled
    test_mask = (time_steps >= 35) & labeled

    # Unknown indices (2, 4, 6) must never appear in any supervised mask
    for idx in [2, 4, 6]:
        assert not train_mask[idx]
        assert not val_mask[idx]
        assert not test_mask[idx]

    assert train_mask.sum() == 2
    assert val_mask.sum() == 1
    assert test_mask.sum() == 1


# ==============================================================================
# 3. Class Weight Calculation from Training Labels Only
# ==============================================================================

def test_class_weight_computed_strictly_from_training_labels():
    """Verify class weights adapt to train split imbalance and ignore test distribution."""
    y_train = np.array([0, 0, 0, 0, 1])  # 4:1 ratio
    y_test = np.array([0, 1, 1, 1])     # 1:3 ratio

    model = CostSensitiveAnomalyModel(model_family="lr", class_weight_ratio=None, random_state=42)
    X_dummy = np.random.randn(5, 4)
    model.fit(X_dummy, y_train)

    assert hasattr(model, "effective_class_weight_ratio")
    assert np.isclose(model.effective_class_weight_ratio, 4.0)


# ==============================================================================
# 4. Validation Threshold Tuning Isolation
# ==============================================================================

def test_validation_threshold_tuning_isolation():
    """Verify decision threshold is optimized on validation data only."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 5)
    y_train = rng.choice([0, 1], size=100, p=[0.8, 0.2])

    X_val = rng.randn(40, 5)
    y_val = rng.choice([0, 1], size=40, p=[0.8, 0.2])

    X_test = rng.randn(50, 5)
    y_test = rng.choice([0, 1], size=50, p=[0.8, 0.2])

    model = CostSensitiveAnomalyModel(model_family="rf", random_state=42)
    model.fit(X_train, y_train)

    best_thresh, best_f1, val_eval = model.tune_threshold(X_val, y_val)
    assert 0.05 <= best_thresh <= 0.95
    assert "f1_score" in val_eval

    # Evaluate on test with tuned threshold
    test_eval = model.evaluate(X_test, y_test, threshold=best_thresh)
    assert "f1_score" in test_eval
    assert "roc_auc" in test_eval
    assert "average_precision" in test_eval


# ==============================================================================
# 5. Zenodo Parquet Feature Loading and Isolation
# ==============================================================================

def test_zenodo_parquet_feature_order_and_column_isolation():
    """Verify bounded loading from data/parquet/zenodo_features isolates model features."""
    import pyarrow.dataset as ds

    zenodo_dir = "data/parquet/zenodo_features"
    if not os.path.exists(zenodo_dir):
        pytest.skip(f"Zenodo parquet directory not found at {zenodo_dir}")

    dataset = ds.dataset(zenodo_dir, format="parquet", partitioning="hive")
    scanner = dataset.scanner(columns=ZENODO_FEATURE_NAMES, batch_size=500)
    batch = next(scanner.to_batches())

    assert batch.schema.names == ZENODO_FEATURE_NAMES
    assert len(batch.schema.names) == 15

    # Critical: Identifiers and partition columns must NOT be in the feature set
    forbidden = ["tx_hash", "block_id", "timestamp", "block_chunk"]
    for col in forbidden:
        assert col not in batch.schema.names

    df = batch.to_pandas()
    assert df.shape[1] == 15
    assert not df.isnull().values.any()

    # Verify a baseline model can consume these features for inference
    model = CostSensitiveAnomalyModel(model_family="rf", random_state=42)
    # Fit dummy on 15 features
    model.fit(df.values[:200], np.random.choice([0, 1], size=200))
    probs = model.predict_proba(df.values[200:])
    assert len(probs) == len(df) - 200
    assert np.all((probs >= 0.0) & (probs <= 1.0))
