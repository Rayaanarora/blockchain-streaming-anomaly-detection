"""
Elliptic Bitcoin Transaction Dataset Preprocessing and Validation Pipeline.

Provides schema validation, strict ID-based alignment, temporal graph partitioning,
and train-only feature transformation to prevent future data leakage.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ELLIPTIC_NUM_FEATURES = 165
FEATURE_COL_NAMES = [f"f{i}" for i in range(1, ELLIPTIC_NUM_FEATURES + 1)]


@dataclass
class PreprocessorConfig:
    train_cutoff_step: int = 34
    val_cutoff_step: Optional[int] = None  # E.g. steps 30-34 as validation if desired
    fill_missing_value: float = 0.0
    scale_features: bool = True
    random_state: int = 42


@dataclass
class PreprocessedDataset:
    tx_ids: np.ndarray                 # Aligned transaction IDs
    time_steps: np.ndarray             # Discrete time step per node (1..49)
    X_raw: np.ndarray                  # Unscaled feature matrix (N, 165)
    X_scaled: np.ndarray               # Scaled feature matrix (fitted strictly on train)
    y: np.ndarray                      # Integer labels: 1=illicit, 0=licit, -1=unknown
    train_mask: np.ndarray             # Boolean mask for supervised training
    val_mask: Optional[np.ndarray]     # Boolean mask for validation (if configured)
    test_mask: np.ndarray              # Boolean mask for inductive testing
    edge_index: np.ndarray             # Directed edges (2, E) aligned to node indices
    edge_time_steps: np.ndarray        # Edge time steps (edges within or across steps)
    node_to_idx: Dict[int, int]        # Map from tx_id to array index


@dataclass
class PreprocessingAuditReport:
    total_raw_feature_rows: int = 0
    total_raw_class_rows: int = 0
    total_raw_edge_rows: int = 0
    feature_cols_detected: int = 0
    duplicate_tx_ids_features: int = 0
    duplicate_tx_ids_classes: int = 0
    duplicate_edges: int = 0
    missing_values_features: int = 0
    non_finite_values_features: int = 0
    unmatched_feature_ids: int = 0
    unmatched_class_ids: int = 0
    unmatched_edge_endpoints: int = 0
    cross_timestep_edges: int = 0
    licit_count: int = 0
    illicit_count: int = 0
    unknown_count: int = 0
    invalid_label_count: int = 0
    time_step_range: Tuple[int, int] = (0, 0)
    train_nodes_count: int = 0
    val_nodes_count: int = 0
    test_nodes_count: int = 0
    train_licit_count: int = 0
    train_illicit_count: int = 0
    test_licit_count: int = 0
    test_illicit_count: int = 0
    exclusions_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EllipticPreprocessor:
    """
    Robust preprocessor for the Elliptic transaction dataset.
    Enforces strict relational joining, chronological integrity, and isolation of unknowns.
    """
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        self.scaler: Optional[StandardScaler] = None

    def audit_and_preprocess(
        self,
        features_df: pd.DataFrame,
        classes_df: pd.DataFrame,
        edgelist_df: pd.DataFrame
    ) -> Tuple[PreprocessedDataset, PreprocessingAuditReport]:
        """
        Validate, align, and split Elliptic data components.
        """
        report = PreprocessingAuditReport()

        report.total_raw_feature_rows = len(features_df)
        report.total_raw_class_rows = len(classes_df)
        report.total_raw_edge_rows = len(edgelist_df)
        report.feature_cols_detected = features_df.shape[1]

        # ----------------------------------------------------------------------
        # 1. Schema & Header Normalization
        # ----------------------------------------------------------------------
        # Features DataFrame: expected 167 columns (tx_id, time_step, f1..f165)
        # Kaggle files have NO header for features
        feat_df = features_df.copy()
        if feat_df.shape[1] == 167:
            feat_df.columns = ["tx_id", "time_step"] + FEATURE_COL_NAMES
        elif feat_df.shape[1] > 2:
            # Handle potential header present in alternative mirrors
            cols = list(feat_df.columns)
            feat_df.rename(columns={cols[0]: "tx_id", cols[1]: "time_step"}, inplace=True)
            feature_names = [f"f{i}" for i in range(1, feat_df.shape[1] - 1)]
            feat_df.columns = ["tx_id", "time_step"] + feature_names
        else:
            raise ValueError(f"Features DataFrame has invalid column count: {feat_df.shape[1]}")

        # Classes DataFrame: expected (tx_id, class)
        cls_df = classes_df.copy()
        cls_cols = list(cls_df.columns)
        if len(cls_cols) >= 2:
            cls_df.rename(columns={cls_cols[0]: "tx_id", cls_cols[1]: "class"}, inplace=True)
        else:
            raise ValueError(f"Classes DataFrame has fewer than 2 columns: {cls_cols}")

        # Edgelist DataFrame: expected (txId1, txId2)
        edge_df = edgelist_df.copy()
        edge_cols = list(edge_df.columns)
        if len(edge_cols) >= 2:
            edge_df.rename(columns={edge_cols[0]: "source", edge_cols[1]: "target"}, inplace=True)
        else:
            raise ValueError(f"Edgelist DataFrame has fewer than 2 columns: {edge_cols}")

        # ----------------------------------------------------------------------
        # 2. Duplicate Detection & ID Hygiene
        # ----------------------------------------------------------------------
        # Check duplicate tx_ids in features
        dup_feat = feat_df["tx_id"].duplicated()
        report.duplicate_tx_ids_features = int(dup_feat.sum())
        if report.duplicate_tx_ids_features > 0:
            dup_ids = feat_df.loc[dup_feat, "tx_id"].tolist()
            report.exclusions_log.append({
                "reason": "duplicate_tx_id_in_features",
                "count": report.duplicate_tx_ids_features,
                "sample_ids": dup_ids[:5]
            })
            feat_df = feat_df.drop_duplicates(subset=["tx_id"], keep="first")

        # Check duplicate tx_ids in classes
        dup_cls = cls_df["tx_id"].duplicated()
        report.duplicate_tx_ids_classes = int(dup_cls.sum())
        if report.duplicate_tx_ids_classes > 0:
            dup_ids = cls_df.loc[dup_cls, "tx_id"].tolist()
            report.exclusions_log.append({
                "reason": "duplicate_tx_id_in_classes",
                "count": report.duplicate_tx_ids_classes,
                "sample_ids": dup_ids[:5]
            })
            cls_df = cls_df.drop_duplicates(subset=["tx_id"], keep="first")

        # Check duplicate edges
        dup_edges = edge_df.duplicated(subset=["source", "target"])
        report.duplicate_edges = int(dup_edges.sum())
        if report.duplicate_edges > 0:
            report.exclusions_log.append({
                "reason": "duplicate_edges",
                "count": report.duplicate_edges
            })
            edge_df = edge_df.drop_duplicates(subset=["source", "target"], keep="first")

        # ----------------------------------------------------------------------
        # 3. Label Validation & Encoding (Unknown Isolation)
        # ----------------------------------------------------------------------
        # In Elliptic:
        # '1' = illicit
        # '2' = licit
        # 'unknown' = unlabeled background transaction
        # NEVER convert 'unknown' to licit (0)!
        cls_df["class_str"] = cls_df["class"].astype(str).str.strip().str.lower()
        
        valid_labels = {"1", "2", "unknown"}
        invalid_mask = ~cls_df["class_str"].isin(valid_labels)
        report.invalid_label_count = int(invalid_mask.sum())
        if report.invalid_label_count > 0:
            invalid_ids = cls_df.loc[invalid_mask, "tx_id"].tolist()
            report.exclusions_log.append({
                "reason": "invalid_label_string",
                "count": report.invalid_label_count,
                "sample_ids": invalid_ids[:5]
            })
            cls_df.loc[invalid_mask, "class_str"] = "unknown"

        label_map = {"1": 1, "2": 0, "unknown": -1}
        cls_df["label_enc"] = cls_df["class_str"].map(label_map).fillna(-1).astype(int)

        report.illicit_count = int((cls_df["label_enc"] == 1).sum())
        report.licit_count = int((cls_df["label_enc"] == 0).sum())
        report.unknown_count = int((cls_df["label_enc"] == -1).sum())

        # ----------------------------------------------------------------------
        # 4. Explicit Relational Alignment by Transaction ID
        # ----------------------------------------------------------------------
        feat_ids: Set[int] = set(feat_df["tx_id"])
        cls_ids: Set[int] = set(cls_df["tx_id"])

        report.unmatched_feature_ids = len(feat_ids - cls_ids)
        report.unmatched_class_ids = len(cls_ids - feat_ids)

        if report.unmatched_feature_ids > 0:
            report.exclusions_log.append({
                "reason": "feature_id_not_in_classes",
                "count": report.unmatched_feature_ids
            })

        if report.unmatched_class_ids > 0:
            report.exclusions_log.append({
                "reason": "class_id_not_in_features",
                "count": report.unmatched_class_ids
            })

        # Inner merge ensures rigorous row-to-row alignment
        aligned_df = feat_df.merge(cls_df[["tx_id", "label_enc"]], on="tx_id", how="inner")
        
        # Sort chronologically by time_step, then tx_id to ensure determinism
        aligned_df = aligned_df.sort_values(by=["time_step", "tx_id"]).reset_index(drop=True)

        # ----------------------------------------------------------------------
        # 5. Numerical Feature Validation (Missing & Non-Finite)
        # ----------------------------------------------------------------------
        feature_cols = [c for c in aligned_df.columns if c not in ("tx_id", "time_step", "label_enc")]
        X_raw = aligned_df[feature_cols].values.astype(np.float32)

        nan_mask = np.isnan(X_raw)
        inf_mask = np.isinf(X_raw)
        report.missing_values_features = int(nan_mask.sum())
        report.non_finite_values_features = int(inf_mask.sum())

        if report.missing_values_features > 0 or report.non_finite_values_features > 0:
            report.exclusions_log.append({
                "reason": "imputed_non_finite_features",
                "nan_count": report.missing_values_features,
                "inf_count": report.non_finite_values_features,
                "fill_value": self.config.fill_missing_value
            })
            X_raw = np.nan_to_num(X_raw, nan=self.config.fill_missing_value, posinf=1e6, neginf=-1e6)

        tx_ids = aligned_df["tx_id"].values
        time_steps = aligned_df["time_step"].values.astype(int)
        y = aligned_df["label_enc"].values.astype(int)

        report.time_step_range = (int(time_steps.min()), int(time_steps.max()))

        # ----------------------------------------------------------------------
        # 6. Graph Edgelist Alignment & Endpoint Validation
        # ----------------------------------------------------------------------
        node_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
        node_timestep_map = {tx_ids[i]: time_steps[i] for i in range(len(tx_ids))}

        # Filter edges where endpoints exist in aligned node set
        valid_src = edge_df["source"].isin(node_to_idx)
        valid_dst = edge_df["target"].isin(node_to_idx)
        valid_edge_mask = valid_src & valid_dst

        report.unmatched_edge_endpoints = int((~valid_edge_mask).sum())
        if report.unmatched_edge_endpoints > 0:
            report.exclusions_log.append({
                "reason": "edge_endpoint_not_in_aligned_nodes",
                "count": report.unmatched_edge_endpoints
            })

        valid_edges = edge_df[valid_edge_mask].copy()

        # Check cross-timestep edges (Elliptic transactions should exist within the same time step)
        src_times = valid_edges["source"].map(node_timestep_map).values
        dst_times = valid_edges["target"].map(node_timestep_map).values
        cross_mask = (src_times != dst_times)
        report.cross_timestep_edges = int(cross_mask.sum())

        if report.cross_timestep_edges > 0:
            report.exclusions_log.append({
                "reason": "cross_timestep_edges_detected",
                "count": report.cross_timestep_edges,
                "note": "Edges connecting different time steps violate strict intra-step snapshot assumption."
            })

        edge_src_idx = valid_edges["source"].map(node_to_idx).values
        edge_dst_idx = valid_edges["target"].map(node_to_idx).values
        edge_index = np.vstack([edge_src_idx, edge_dst_idx])
        edge_time_steps = src_times  # Using source time step as primary timestamp

        # ----------------------------------------------------------------------
        # 7. Temporal Split & Train-Only Feature Scaling
        # ----------------------------------------------------------------------
        cutoff = self.config.train_cutoff_step
        
        # Only labeled nodes (y != -1) participate in supervised loss / metric evaluation
        labeled_mask = (y != -1)
        
        if self.config.val_cutoff_step is not None:
            val_cutoff = self.config.val_cutoff_step
            train_mask = (time_steps <= val_cutoff) & labeled_mask
            val_mask = (time_steps > val_cutoff) & (time_steps <= cutoff) & labeled_mask
            test_mask = (time_steps > cutoff) & labeled_mask
        else:
            train_mask = (time_steps <= cutoff) & labeled_mask
            val_mask = None
            test_mask = (time_steps > cutoff) & labeled_mask

        report.train_nodes_count = int(train_mask.sum())
        report.val_nodes_count = int(val_mask.sum()) if val_mask is not None else 0
        report.test_nodes_count = int(test_mask.sum())

        report.train_licit_count = int(((y == 0) & train_mask).sum())
        report.train_illicit_count = int(((y == 1) & train_mask).sum())
        report.test_licit_count = int(((y == 0) & test_mask).sum())
        report.test_illicit_count = int(((y == 1) & test_mask).sum())

        # Fit StandardScaler STRICTLY on the training nodes!
        # CRITICAL LEAKAGE PREVENTION: Scaler parameters (mean, std) must not see validation or test sets.
        if self.config.scale_features:
            self.scaler = StandardScaler()
            # Fit on training features only
            self.scaler.fit(X_raw[train_mask])
            X_scaled = self.scaler.transform(X_raw)
        else:
            X_scaled = X_raw.copy()

        dataset = PreprocessedDataset(
            tx_ids=tx_ids,
            time_steps=time_steps,
            X_raw=X_raw,
            X_scaled=X_scaled,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            edge_index=edge_index,
            edge_time_steps=edge_time_steps,
            node_to_idx=node_to_idx
        )

        return dataset, report

    @staticmethod
    def generate_markdown_audit_report(report: PreprocessingAuditReport, dataset_summary: Optional[Dict[str, Any]] = None) -> str:
        """Format an audit report into Markdown."""
        return f"""# Elliptic Benchmark Preprocessing Audit Report

## 1. Raw Dataset Ingestion Summary
- **Raw Feature Rows:** {report.total_raw_feature_rows:,}
- **Raw Class Rows:** {report.total_raw_class_rows:,}
- **Raw Edge Rows:** {report.total_raw_edge_rows:,}
- **Feature Dimension:** {report.feature_cols_detected} columns detected (165 features + tx_id + time_step)

## 2. Data Hygiene & Integrity Checks
- **Duplicate Tx IDs in Features:** {report.duplicate_tx_ids_features}
- **Duplicate Tx IDs in Classes:** {report.duplicate_tx_ids_classes}
- **Duplicate Edges:** {report.duplicate_edges}
- **Missing / NaN Feature Values:** {report.missing_values_features}
- **Non-Finite / Inf Feature Values:** {report.non_finite_values_features}
- **Unmatched Feature IDs:** {report.unmatched_feature_ids}
- **Unmatched Class IDs:** {report.unmatched_class_ids}
- **Unmatched Edge Endpoints:** {report.unmatched_edge_endpoints}
- **Cross-Timestep Edges:** {report.cross_timestep_edges}

## 3. Ground Truth Label Distribution
- **Licit Transactions (Class 2 -> 0):** {report.licit_count:,}
- **Illicit Transactions (Class 1 -> 1):** {report.illicit_count:,}
- **Unknown Transactions (-1):** {report.unknown_count:,} (Preserved separately; **NEVER** collapsed into licit)
- **Invalid Labels Found:** {report.invalid_label_count}

## 4. Chronological Inductive Partitioning
- **Time Step Range:** Steps {report.time_step_range[0]} through {report.time_step_range[1]}
- **Training Set (Steps 1-{report.time_step_range[0] if report.train_nodes_count == 0 else 34}):**
  - Total Labeled Nodes: {report.train_nodes_count:,}
  - Licit: {report.train_licit_count:,} | Illicit: {report.train_illicit_count:,}
  - Class Imbalance Ratio: {f"{report.train_licit_count / max(report.train_illicit_count, 1):.2f}:1" if report.train_illicit_count > 0 else "N/A"}
- **Testing Set (Steps 35-49):**
  - Total Labeled Nodes: {report.test_nodes_count:,}
  - Licit: {report.test_licit_count:,} | Illicit: {report.test_illicit_count:,}
  - Class Imbalance Ratio: {f"{report.test_licit_count / max(report.test_illicit_count, 1):.2f}:1" if report.test_illicit_count > 0 else "N/A"}

## 5. Methodological & Leakage Prevention Guarantees
- **Train-Only Scaling:** `StandardScaler` fitted strictly on training partition (`time_step <= 34` and `labeled`), preventing future-to-past feature leakage.
- **Explicit Relational Alignment:** Rows aligned strictly on `tx_id` join rather than positional index.
- **Unknown Isolation:** 157k background transactions retained for graph topology but excluded from supervised loss.

## 6. Exclusions and Anomalies Log
```json
{json.dumps(report.exclusions_log, indent=2)}
```
"""
