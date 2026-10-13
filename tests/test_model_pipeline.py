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


# ==============================================================================
# 6. Evaluation Artifacts Consistency & Temporal Drift
# ==============================================================================

def test_saved_predictions_and_metrics_consistency():
    """Verify test_predictions.parquet matches metrics.json confusion matrix and metrics."""
    import json
    pred_path = "data/validation/model_evaluation/test_predictions.parquet"
    metrics_path = "data/validation/model_evaluation/metrics.json"

    if not os.path.exists(pred_path) or not os.path.exists(metrics_path):
        pytest.skip("Evaluation artifacts not found")

    df_preds = pd.read_parquet(pred_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert len(df_preds) == 16670
    assert "y_true" in df_preds.columns

    for model_key in ["lr", "rf", "gbdt"]:
        prob_col = f"prob_{model_key}"
        pred_col = f"pred_{model_key}_thresh_selected"
        assert prob_col in df_preds.columns
        assert pred_col in df_preds.columns

        model_meta = metrics["models"][model_key]
        thresh = model_meta["selected_threshold"]
        expected_cm = model_meta["test_results_at_selected_threshold"]["confusion_matrix"]

        # Verify saved binary predictions match reported confusion matrix exactly
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
        y_true = df_preds["y_true"].values
        cm = confusion_matrix(y_true, df_preds[pred_col].values)
        np.testing.assert_array_equal(cm, expected_cm)

        # Precision, recall, f1 from saved predictions match metrics.json
        p = precision_score(y_true, df_preds[pred_col].values, pos_label=1)
        r = recall_score(y_true, df_preds[pred_col].values, pos_label=1)
        f1 = f1_score(y_true, df_preds[pred_col].values, pos_label=1)

        reported_p = model_meta["test_results_at_selected_threshold"]["precision"]
        reported_r = model_meta["test_results_at_selected_threshold"]["recall"]
        reported_f1 = model_meta["test_results_at_selected_threshold"]["f1_score"]

        assert abs(p - reported_p) < 1e-4
        assert abs(r - reported_r) < 1e-4
        assert abs(f1 - reported_f1) < 1e-4

        # Verify probabilities match predictions (>99.99% exact match, accounting for 4-decimal rounding in saved float)
        recomputed_pred = (df_preds[prob_col] >= thresh).astype(int)
        agreement = np.mean(df_preds[pred_col].values == recomputed_pred.values)
        assert agreement >= 0.9999, f"Expected probability-to-pred agreement >= 99.99%, got {agreement}"


def test_temporal_drift_concept_shift_step_43():
    """Verify that temporal performance shift occurs at step 43 (AlphaBay shutdown)."""
    pred_path = "data/validation/model_evaluation/test_predictions.parquet"
    if not os.path.exists(pred_path):
        pytest.skip("Evaluation predictions not found")

    df_preds = pd.read_parquet(pred_path)
    assert "time_step" in df_preds.columns

    from sklearn.metrics import f1_score
    # Pre-shutdown: steps 35-42
    pre = df_preds[df_preds["time_step"] <= 42]
    # Post-shutdown: steps 43-49
    post = df_preds[df_preds["time_step"] >= 43]

    # Pre-shutdown RF F1 should be high (> 0.75), post-shutdown should collapse (< 0.10)
    rf_f1_pre = f1_score(pre["y_true"], pre["pred_rf_thresh_selected"], pos_label=1)
    rf_f1_post = f1_score(post["y_true"], post["pred_rf_thresh_selected"], pos_label=1)

    assert rf_f1_pre > 0.75, f"Expected pre-shutdown RF F1 > 0.75, got {rf_f1_pre}"
    assert rf_f1_post < 0.10, f"Expected post-shutdown RF F1 < 0.10, got {rf_f1_post}"


def test_latency_benchmark_contract():
    """Verify reproducible latency benchmark report exists and satisfies contract."""
    import json
    bench_path = "data/validation/model_evaluation/latency_benchmark.json"
    if not os.path.exists(bench_path):
        pytest.skip("Latency benchmark report not found")

    with open(bench_path, "r") as f:
        data = json.load(f)

    assert "hardware_environment" in data
    assert "models" in data
    assert "timer" in data["hardware_environment"]

    for model_key in ["lr", "rf", "gbdt"]:
        assert model_key in data["models"]
        model_metrics = data["models"][model_key]["latency_metrics"]
        assert "batch_1" in model_metrics
        assert "batch_1000" in model_metrics
        assert "batch_16670" in model_metrics

        b1 = model_metrics["batch_1"]["batch_latency_ms"]
        assert "median_p50" in b1
        assert "p95" in b1
        assert b1["median_p50"] > 0
        assert b1["p95"] >= b1["median_p50"]

