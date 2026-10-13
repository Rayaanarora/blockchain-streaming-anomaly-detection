"""
Synthetic unit and integration tests for Elliptic dataset preprocessing.

Verifies:
- Strict relational alignment by transaction ID
- Duplicate detection and invalid label handling
- Missing and non-finite feature imputation
- Invalid graph endpoint rejection
- Unknown-label separation (never collapsed to licit)
- Temporal inductive split integrity
- Train-only feature scaling (zero test-set leakage)
- Cross-timestep edge detection
- Deterministic preprocessing outputs
"""
import os
import pytest
import numpy as np
import pandas as pd
from src.data.elliptic import (
    EllipticPreprocessor,
    PreprocessorConfig,
    PreprocessedDataset,
    PreprocessingAuditReport,
    ELLIPTIC_NUM_FEATURES,
    ELLIPTIC_NUM_LOCAL_FEATURES,
    ELLIPTIC_NUM_AGG_FEATURES,
)


def create_synthetic_elliptic_data(num_samples: int = 20):
    """Generate small synthetic Elliptic dataframes with 165 features."""
    tx_ids = np.arange(1001, 1001 + num_samples)
    # Distribute across time steps 1 to 40
    time_steps = np.random.RandomState(42).choice(np.arange(1, 41), size=num_samples)
    
    # 165 features
    feature_matrix = np.random.RandomState(42).randn(num_samples, ELLIPTIC_NUM_FEATURES).astype(np.float32)
    features_df = pd.DataFrame(feature_matrix)
    features_df.insert(0, "time_step", time_steps)
    features_df.insert(0, "tx_id", tx_ids)

    # Classes: mix of licit ('2'), illicit ('1'), and unknown ('unknown')
    classes = ["2", "1", "unknown", "2"] * (num_samples // 4 + 1)
    classes = classes[:num_samples]
    classes_df = pd.DataFrame({"tx_id": tx_ids, "class": classes})

    # Edgelist: directed intra-step edges
    sources = tx_ids[:num_samples // 2]
    targets = tx_ids[num_samples // 2:num_samples // 2 + len(sources)]
    edgelist_df = pd.DataFrame({"txId1": sources, "txId2": targets})

    return features_df, classes_df, edgelist_df


# ==============================================================================
# 1. Feature / Label ID Alignment
# ==============================================================================

def test_feature_label_id_alignment():
    """Verify alignment relies strictly on transaction ID, never row position."""
    features_df, classes_df, edgelist_df = create_synthetic_elliptic_data(10)

    # Scramble row orders independently
    features_scrambled = features_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    classes_scrambled = classes_df.sample(frac=1.0, random_state=456).reset_index(drop=True)

    preprocessor = EllipticPreprocessor()
    dataset, report = preprocessor.audit_and_preprocess(features_scrambled, classes_scrambled, edgelist_df)

    # Verify every single row in dataset.X_raw aligns with dataset.y for that exact tx_id
    id_to_orig_label = dict(zip(classes_df["tx_id"], classes_df["class"]))
    label_map = {"1": 1, "2": 0, "unknown": -1}

    for idx, tx_id in enumerate(dataset.tx_ids):
        orig_class_str = id_to_orig_label[tx_id]
        expected_label = label_map[orig_class_str]
        assert dataset.y[idx] == expected_label


# ==============================================================================
# 2. Duplicate IDs and Invalid Labels
# ==============================================================================

def test_duplicate_ids_and_invalid_labels():
    """Verify duplicate IDs are safely deduplicated and invalid label strings quarantined."""
    features_df, classes_df, edgelist_df = create_synthetic_elliptic_data(8)

    # Inject duplicate tx_id into features
    dup_row = features_df.iloc[[0]].copy()
    features_df = pd.concat([features_df, dup_row], ignore_index=True)

    # Inject invalid label string
    classes_df.loc[1, "class"] = "malicious_ransomware_string"

    preprocessor = EllipticPreprocessor()
    dataset, report = preprocessor.audit_and_preprocess(features_df, classes_df, edgelist_df)

    assert report.duplicate_tx_ids_features == 1
    assert report.invalid_label_count == 1
    # Invalid label converted to -1 (unknown) rather than licit (0)
    assert dataset.y[dataset.node_to_idx[classes_df.loc[1, "tx_id"]]] == -1
    assert len(report.exclusions_log) >= 2


# ==============================================================================
# 3. Missing and Non-Finite Features
# ==============================================================================

def test_missing_and_non_finite_features():
    """Verify NaNs and Infs are detected, logged, and imputed."""
    features_df, classes_df, edgelist_df = create_synthetic_elliptic_data(6)

    # Inject NaN and Inf into features (col index 2 is f1)
    features_df.iloc[0, 2] = np.nan
    features_df.iloc[1, 3] = np.inf
    features_df.iloc[2, 4] = -np.inf

    preprocessor = EllipticPreprocessor()
    dataset, report = preprocessor.audit_and_preprocess(features_df, classes_df, edgelist_df)

    assert report.missing_values_features == 1
    assert report.non_finite_values_features == 2
    assert not np.isnan(dataset.X_raw).any()
    assert not np.isinf(dataset.X_raw).any()


# ==============================================================================
# 4. Invalid Graph Endpoints
# ==============================================================================

def test_invalid_graph_endpoints():
    """Verify edges pointing to non-existent transaction IDs are quarantined."""
    features_df, classes_df, edgelist_df = create_synthetic_elliptic_data(6)

    # Add phantom edge with non-existent source and target
    phantom_edges = pd.DataFrame({"txId1": [999999, 1001], "txId2": [1002, 888888]})
    edgelist_df = pd.concat([edgelist_df, phantom_edges], ignore_index=True)

    preprocessor = EllipticPreprocessor()
    dataset, report = preprocessor.audit_and_preprocess(features_df, classes_df, edgelist_df)

    assert report.unmatched_edge_endpoints == 2
    # Ensure edge_index contains only valid indices in [0, len(nodes) - 1]
    assert dataset.edge_index[0].max() < len(dataset.tx_ids)
    assert dataset.edge_index[1].max() < len(dataset.tx_ids)


# ==============================================================================
# 5. Unknown-Label Handling
# ==============================================================================

def test_unknown_label_isolation():
    """Verify unknown labels are encoded as -1 and NEVER collapsed into licit class (0)."""
    features_df, classes_df, edgelist_df = create_synthetic_elliptic_data(10)

    # Ensure known counts
    classes_df["class"] = ["unknown"] * 5 + ["2"] * 3 + ["1"] * 2

    preprocessor = EllipticPreprocessor()
    dataset, report = preprocessor.audit_and_preprocess(features_df, classes_df, edgelist_df)

    assert report.unknown_count == 5
    assert report.licit_count == 3
    assert report.illicit_count == 2

    # Verify unknown nodes have y == -1
    unknown_indices = np.where(dataset.y == -1)[0]
    assert len(unknown_indices) == 5

    # Unknown nodes MUST NOT be in train_mask or test_mask
    assert not dataset.train_mask[unknown_indices].any()
    assert not dataset.test_mask[unknown_indices].any()


# ==============================================================================
# 6. Temporal Split Integrity
# ==============================================================================

def test_temporal_split_integrity():
    """Verify chronological split partition strictly enforces cutoff threshold."""
    features_df, classes_df, edgelist_df = create_synthetic_elliptic_data(20)

    # Ensure labeled classes across time steps 1..40
    classes_df["class"] = ["1", "2"] * 10
    features_df["time_step"] = np.repeat([10, 20, 30, 34, 35, 36, 40], [3, 3, 3, 3, 3, 3, 2])

    config = PreprocessorConfig(train_cutoff_step=34)
    preprocessor = EllipticPreprocessor(config=config)
    dataset, report = preprocessor.audit_and_preprocess(features_df, classes_df, edgelist_df)

    # Train nodes must have time_step <= 34
    train_times = dataset.time_steps[dataset.train_mask]
    assert (train_times <= 34).all()

    # Test nodes must have time_step > 34
    test_times = dataset.time_steps[dataset.test_mask]
    assert (test_times > 34).all()

    # Overlap must be empty
    overlap = dataset.train_mask & dataset.test_mask
    assert not overlap.any()


# ==============================================================================
# 7. Train-Only Scaling (Zero Future Leakage)
# ==============================================================================

def test_train_only_scaling_zero_leakage():
    """
    Verify feature scaling is fitted strictly on training partition.
    Drastic changes in future test nodes must NOT affect scaled values of training nodes.
    """
    features_df1, classes_df, edgelist_df = create_synthetic_elliptic_data(10)
    classes_df["class"] = ["1"] * 5 + ["2"] * 5
    # Steps 1..5 are train (<=34), steps 35..39 are test (>34)
    features_df1["time_step"] = [10, 15, 20, 25, 30, 35, 36, 37, 38, 39]

    # Preprocess dataset 1
    p1 = EllipticPreprocessor(config=PreprocessorConfig(train_cutoff_step=34))
    d1, _ = p1.audit_and_preprocess(features_df1, classes_df, edgelist_df)

    # Create dataset 2 with extreme outlier injected ONLY in the test partition (step 39)
    features_df2 = features_df1.copy()
    features_df2.iloc[9, 2:] = 1e9  # Massive outlier in future node

    p2 = EllipticPreprocessor(config=PreprocessorConfig(train_cutoff_step=34))
    d2, _ = p2.audit_and_preprocess(features_df2, classes_df, edgelist_df)

    # The scaled features of all TRAINING nodes must be 100% IDENTICAL
    train_idx1 = np.where(d1.train_mask)[0]
    train_idx2 = np.where(d2.train_mask)[0]
    np.testing.assert_allclose(d1.X_scaled[train_idx1], d2.X_scaled[train_idx2], rtol=1e-5)


# ==============================================================================
# 8. Cross-Timestep Edge Detection
# ==============================================================================

def test_cross_timestep_edge_detection():
    """Verify edges connecting nodes across different time steps are flagged."""
    features_df, classes_df, _ = create_synthetic_elliptic_data(4)
    features_df["time_step"] = [1, 1, 2, 2]

    # Edge 1 is intra-step (1 -> 1), Edge 2 is cross-step (1 -> 2)
    cross_edges = pd.DataFrame({
        "txId1": [features_df["tx_id"].iloc[0], features_df["tx_id"].iloc[1]],
        "txId2": [features_df["tx_id"].iloc[1], features_df["tx_id"].iloc[2]]
    })

    preprocessor = EllipticPreprocessor()
    _, report = preprocessor.audit_and_preprocess(features_df, classes_df, cross_edges)

    assert report.cross_timestep_edges == 1
    assert any(x["reason"] == "cross_timestep_edges_detected" for x in report.exclusions_log)


# ==============================================================================
# 9. Deterministic Preprocessing
# ==============================================================================

def test_deterministic_preprocessing():
    """Verify running preprocessing on identical data produces deterministic bitwise outputs."""
    features_df, classes_df, edgelist_df = create_synthetic_elliptic_data(12)

    p1 = EllipticPreprocessor()
    d1, r1 = p1.audit_and_preprocess(features_df, classes_df, edgelist_df)

    p2 = EllipticPreprocessor()
    d2, r2 = p2.audit_and_preprocess(features_df, classes_df, edgelist_df)

    np.testing.assert_array_equal(d1.tx_ids, d2.tx_ids)
    np.testing.assert_array_equal(d1.time_steps, d2.time_steps)
    np.testing.assert_allclose(d1.X_scaled, d2.X_scaled)
    np.testing.assert_array_equal(d1.y, d2.y)
    np.testing.assert_array_equal(d1.edge_index, d2.edge_index)


# ==============================================================================
# 10. Persistence & Round-Trip Loading
# ==============================================================================

def test_save_and_load_preprocessed_dataset(tmp_path):
    """Verify preprocessed dataset serialization and deserialization roundtrip."""
    features_df, classes_df, edgelist_df = create_synthetic_elliptic_data(15)
    preprocessor = EllipticPreprocessor()
    dataset, _ = preprocessor.audit_and_preprocess(features_df, classes_df, edgelist_df)

    save_dir = str(tmp_path / "elliptic_processed")
    saved_paths = dataset.save(save_dir)

    assert os.path.exists(saved_paths["npz"])
    assert os.path.exists(saved_paths["features_parquet"])
    assert os.path.exists(saved_paths["edgelist_parquet"])

    # Verify Parquet tables can be read
    df_feat = pd.read_parquet(saved_paths["features_parquet"])
    assert len(df_feat) == len(dataset.tx_ids)
    assert "tx_id" in df_feat.columns
    assert "label" in df_feat.columns
    assert "f1" in df_feat.columns

    df_edges = pd.read_parquet(saved_paths["edgelist_parquet"])
    assert len(df_edges) == dataset.edge_index.shape[1]
    assert "source_tx_id" in df_edges.columns
    assert "target_tx_id" in df_edges.columns

    # Verify round-trip load
    loaded = PreprocessedDataset.load(save_dir)
    np.testing.assert_array_equal(loaded.tx_ids, dataset.tx_ids)
    np.testing.assert_array_equal(loaded.time_steps, dataset.time_steps)
    np.testing.assert_allclose(loaded.X_scaled, dataset.X_scaled)
    np.testing.assert_array_equal(loaded.y, dataset.y)
    np.testing.assert_array_equal(loaded.train_mask, dataset.train_mask)
    np.testing.assert_array_equal(loaded.test_mask, dataset.test_mask)
    np.testing.assert_array_equal(loaded.edge_index, dataset.edge_index)
    np.testing.assert_array_equal(loaded.edge_time_steps, dataset.edge_time_steps)


# ==============================================================================
# 11. Feature Taxonomy & Dimensionality Invariance
# ==============================================================================

def test_elliptic_feature_taxonomy():
    """Verify exact count and partitioning of local vs aggregated features."""
    assert ELLIPTIC_NUM_LOCAL_FEATURES == 93
    assert ELLIPTIC_NUM_AGG_FEATURES == 72
    assert ELLIPTIC_NUM_LOCAL_FEATURES + ELLIPTIC_NUM_AGG_FEATURES == ELLIPTIC_NUM_FEATURES
    assert ELLIPTIC_NUM_FEATURES == 165


