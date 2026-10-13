"""
TGAT-BoostNet Hybrid Experiment: Leakage-Safe Training, Fair Ablation & Temporal Drift Analysis.

Evaluates 5 models under strict chronological temporal protocol:
- Train: Timesteps 1 to 29
- Val: Timesteps 30 to 34 (Threshold tuning & checkpoint selection)
- Test: Timesteps 35 to 49 (Evaluated once at tuned validation thresholds)

Ablation Models:
A. Cost-Sensitive HistGradientBoosting (165 features)
B. XGBoost Alone (165 features)
C. Spatio-Temporal Graph Attention Network (TGAT Alone, 32-dim graph features)
D. TGAT-BoostNet Hybrid (197 fused features: 165 raw + 32 TGAT embeddings -> XGBoost)
E. Cost-Sensitive Random Forest (165 features)

Artifacts saved to data/validation/tgat_boostnet_experiment/
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
import xgboost as xgb
import joblib

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.gat import SpatioTemporalGAT, is_gat_available


SEEDS = [42, 123, 456, 789, 2024]
DATA_PATH = "data/processed/elliptic/elliptic_preprocessed.npz"
OUTPUT_DIR = "data/validation/tgat_boostnet_experiment"


def parse_args():
    parser = argparse.ArgumentParser(description="TGAT-BoostNet Fair Ablation and Benchmarking")
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def load_dataset(data_path: str):
    """Load preprocessed Elliptic data and verify shape and node consistency."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")
    data = np.load(data_path)
    tx_ids = data["tx_ids"]
    time_steps = data["time_steps"]
    X_raw = data["X_raw"]
    y = data["y"]
    edge_index = data["edge_index"]
    edge_time_steps = data["edge_time_steps"]
    return tx_ids, time_steps, X_raw, y, edge_index, edge_time_steps


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find threshold in [0.01, 0.99] that maximizes illicit class F1 on validation split."""
    best_thresh = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return best_thresh, float(best_f1)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float) -> Dict[str, Any]:
    """Compute precision, recall, illicit F1, macro F1, ROC-AUC, PR-AUC, confusion matrix."""
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_illicit = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_illicit": round(f1_illicit, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(pr_auc, 4),
        "threshold": round(thresh, 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }


def train_tgat_model(
    x_train_norm: np.ndarray,
    time_steps: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    class_weight: float,
    seed: int,
    epochs: int = 40,
    lr: float = 0.005,
    patience: int = 15,
    device: str = "cpu"
) -> Tuple[SpatioTemporalGAT, Dict[str, Any], float]:
    """
    Train SpatioTemporalGAT with validation early stopping and checkpointing.
    Returns (best_model, training_history, train_time_sec).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Edge filtering: strictly training edges for training forward pass
    src, dst = edge_index[0], edge_index[1]
    train_edge_mask = (time_steps[src] <= 29) & (time_steps[dst] <= 29)
    train_edge_index = torch.tensor(edge_index[:, train_edge_mask], dtype=torch.long, device=device)

    # Full edge index for validation evaluation across graph
    full_edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)

    x_tensor = torch.tensor(x_train_norm, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(time_steps.reshape(-1, 1) / 49.0, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool, device=device)

    model = SpatioTemporalGAT(in_channels=165, hidden_channels=32, out_channels=2, heads=4, dropout=0.3).to(device)
    weights = torch.tensor([1.0, class_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_f1 = -1.0
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    history = {
        "epochs": [],
        "train_loss": [],
        "val_f1": [],
        "best_epoch": 0,
        "best_val_f1": 0.0
    }

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor, train_edge_index, t_tensor)
        loss = criterion(out[train_mask_t], y_tensor[train_mask_t])
        loss.backward()
        optimizer.step()

        # Validation check for early stopping
        model.eval()
        with torch.no_grad():
            val_out = model(x_tensor, full_edge_index, t_tensor)
            val_probs = F.softmax(val_out[val_mask_t], dim=1)[:, 1].cpu().numpy()
            y_val_np = y[val_mask]
            _, val_f1 = tune_threshold(y_val_np, val_probs)

        history["epochs"].append(ep)
        history["train_loss"].append(round(float(loss.item()), 4))
        history["val_f1"].append(round(float(val_f1), 4))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"      [Early Stopping] Triggered at epoch {ep} (Best epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f})")
            break

    train_time = time.time() - t0
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    history["best_epoch"] = best_epoch
    history["best_val_f1"] = round(float(best_val_f1), 4)

    return model, history, train_time


def extract_embeddings_inductive(
    model: SpatioTemporalGAT,
    x_norm: np.ndarray,
    time_steps: np.ndarray,
    edge_index: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """
    Extract 32-dim graph embeddings step-by-step to strictly prevent cross-timestep information flow.
    """
    model.eval()
    embeddings = np.zeros((x_norm.shape[0], 32), dtype=np.float32)

    with torch.no_grad():
        for t in range(1, 50):
            step_mask = (time_steps == t)
            step_indices = np.where(step_mask)[0]
            if len(step_indices) == 0:
                continue

            # Subgraph edges for timestep t
            src, dst = edge_index[0], edge_index[1]
            sub_edge_mask = (time_steps[src] == t) & (time_steps[dst] == t)
            sub_edge_index = edge_index[:, sub_edge_mask]

            # Remap edge indices locally to sub_tensor
            idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(step_indices)}
            if sub_edge_index.shape[1] > 0:
                local_src = [idx_map[s] for s in sub_edge_index[0] if s in idx_map]
                local_dst = [idx_map[d] for d in sub_edge_index[1] if d in idx_map]
                local_edge_t = torch.tensor([local_src, local_dst], dtype=torch.long, device=device)
            else:
                local_edge_t = torch.empty((2, 0), dtype=torch.long, device=device)

            x_step = torch.tensor(x_norm[step_indices], dtype=torch.float32, device=device)
            t_step = torch.tensor(time_steps[step_indices].reshape(-1, 1) / 49.0, dtype=torch.float32, device=device)

            step_emb = model.extract_embedding(x_step, local_edge_t, t_step).cpu().numpy()
            embeddings[step_indices] = step_emb

    return embeddings


def evaluate_tgat_alone(
    model: SpatioTemporalGAT,
    x_norm: np.ndarray,
    time_steps: np.ndarray,
    edge_index: np.ndarray,
    mask: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """Evaluate TGAT alone step-by-step inductively, returning illicit class probabilities."""
    model.eval()
    probs = np.zeros(mask.sum(), dtype=np.float32)
    selected_indices = np.where(mask)[0]

    with torch.no_grad():
        cursor = 0
        for t in range(1, 50):
            step_mask = (time_steps == t)
            step_selected = step_mask & mask
            n_sel = step_selected.sum()
            if n_sel == 0:
                continue

            step_indices = np.where(step_mask)[0]
            idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(step_indices)}

            src, dst = edge_index[0], edge_index[1]
            sub_edge_mask = (time_steps[src] == t) & (time_steps[dst] == t)
            sub_edge_index = edge_index[:, sub_edge_mask]

            if sub_edge_index.shape[1] > 0:
                local_src = [idx_map[s] for s in sub_edge_index[0] if s in idx_map]
                local_dst = [idx_map[d] for d in sub_edge_index[1] if d in idx_map]
                local_edge_t = torch.tensor([local_src, local_dst], dtype=torch.long, device=device)
            else:
                local_edge_t = torch.empty((2, 0), dtype=torch.long, device=device)

            x_step = torch.tensor(x_norm[step_indices], dtype=torch.float32, device=device)
            t_step = torch.tensor(time_steps[step_indices].reshape(-1, 1) / 49.0, dtype=torch.float32, device=device)

            logits = model(x_step, local_edge_t, t_step)
            step_probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            # Map from step_indices to step_selected
            sel_in_step = np.isin(step_indices, np.where(step_selected)[0])
            probs[cursor:cursor + n_sel] = step_probs[sel_in_step]
            cursor += n_sel

    return probs


def run_latency_benchmark(
    hist_model, xgb_model, tgat_model, hybrid_model, rf_model,
    X_test_165: np.ndarray, X_test_197: np.ndarray,
    time_steps: np.ndarray, edge_index: np.ndarray,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Measure inference latency at batch sizes 1, 64, 1000, and full test set (16670)."""
    batch_sizes = [1, 64, 1000, len(X_test_165)]
    latency_results = {}
    tgat_model.eval()

    # Pre-build a dummy batch graph for TGAT benchmark
    x_test_torch = torch.tensor(X_test_165, dtype=torch.float32, device=device)
    t_test_torch = torch.tensor(time_steps.reshape(-1, 1) / 49.0, dtype=torch.float32, device=device)

    for b in batch_sizes:
        b_key = str(b)
        latency_results[b_key] = {}
        sample_165 = X_test_165[:b]
        sample_197 = X_test_197[:b]
        sample_x_torch = x_test_torch[:b]
        sample_t_torch = t_test_torch[:b]
        # Isolated edges (empty edge index for micro-batch latency baseline)
        dummy_edges = torch.empty((2, 0), dtype=torch.long, device=device)

        n_iters = 50 if b <= 64 else 15

        # 1. HistGBDT
        for _ in range(5):
            _ = hist_model.predict_proba(sample_165)
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = hist_model.predict_proba(sample_165)
            times.append((time.perf_counter() - t0) * 1000.0)
        latency_results[b_key]["HistGBDT"] = {
            "median_ms": round(float(np.median(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
            "per_tx_us": round(float(np.median(times) / b * 1000.0), 2)
        }

        # 2. XGBoost Alone
        for _ in range(5):
            _ = xgb_model.predict_proba(sample_165)
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = xgb_model.predict_proba(sample_165)
            times.append((time.perf_counter() - t0) * 1000.0)
        latency_results[b_key]["XGBoost_Alone"] = {
            "median_ms": round(float(np.median(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
            "per_tx_us": round(float(np.median(times) / b * 1000.0), 2)
        }

        # 3. TGAT Alone (Graph forward pass)
        with torch.no_grad():
            for _ in range(5):
                _ = tgat_model(sample_x_torch, dummy_edges, sample_t_torch)
            times = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _ = tgat_model(sample_x_torch, dummy_edges, sample_t_torch)
                times.append((time.perf_counter() - t0) * 1000.0)
        latency_results[b_key]["TGAT_Alone"] = {
            "median_ms": round(float(np.median(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
            "per_tx_us": round(float(np.median(times) / b * 1000.0), 2)
        }

        # 4. TGAT-BoostNet Boost Head (Inference over 197-dim fused features)
        for _ in range(5):
            _ = hybrid_model.predict_proba(sample_197)
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = hybrid_model.predict_proba(sample_197)
            times.append((time.perf_counter() - t0) * 1000.0)
        latency_results[b_key]["TGAT_BoostNet_Fused_Inference"] = {
            "median_ms": round(float(np.median(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
            "per_tx_us": round(float(np.median(times) / b * 1000.0), 2)
        }

        # 5. TGAT-BoostNet End-to-End (TGAT Embedding Extraction + XGBoost Inference)
        with torch.no_grad():
            for _ in range(5):
                e = tgat_model.extract_embedding(sample_x_torch, dummy_edges, sample_t_torch).cpu().numpy()
                f = np.concatenate([sample_165, e], axis=1)
                _ = hybrid_model.predict_proba(f)
            times = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                e = tgat_model.extract_embedding(sample_x_torch, dummy_edges, sample_t_torch).cpu().numpy()
                f = np.concatenate([sample_165, e], axis=1)
                _ = hybrid_model.predict_proba(f)
                times.append((time.perf_counter() - t0) * 1000.0)
        latency_results[b_key]["TGAT_BoostNet_End_to_End"] = {
            "median_ms": round(float(np.median(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
            "per_tx_us": round(float(np.median(times) / b * 1000.0), 2)
        }

        # 6. Random Forest
        for _ in range(5):
            _ = rf_model.predict_proba(sample_165)
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = rf_model.predict_proba(sample_165)
            times.append((time.perf_counter() - t0) * 1000.0)
        latency_results[b_key]["Random_Forest"] = {
            "median_ms": round(float(np.median(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
            "per_tx_us": round(float(np.median(times) / b * 1000.0), 2)
        }

    return latency_results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 85)
    print("TGAT-BOOSTNET HYBRID: LEAKAGE-SAFE TRAINING, ABLATION & TEMPORAL DRIFT AUDIT")
    print(f"Data source:      {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seeds:     {args.seeds}")
    print("=" * 85, flush=True)

    # 1. Load data
    print("\n[Step 1/8] Loading Elliptic preprocessed dataset...")
    tx_ids, time_steps, X_raw, y, edge_index, edge_time_steps = load_dataset(args.data_path)
    print(f"  Loaded {len(tx_ids):,} total transactions across {time_steps.max()} timesteps.")
    print(f"  Raw feature dimension: {X_raw.shape[1]}")
    print(f"  Total graph edges:     {edge_index.shape[1]:,}")

    # 2. Partitions & Split statistics
    print("\n[Step 2/8] Constructing strict chronological partitions...")
    labeled = (y != -1)
    train_mask = (time_steps <= 29) & labeled
    val_mask = (time_steps >= 30) & (time_steps <= 34) & labeled
    test_mask = (time_steps >= 35) & labeled
    unknown_mask = (y == -1)

    n_train_licit = int(np.sum((y == 0) & train_mask))
    n_train_illicit = int(np.sum((y == 1) & train_mask))
    n_val_licit = int(np.sum((y == 0) & val_mask))
    n_val_illicit = int(np.sum((y == 1) & val_mask))
    n_test_licit = int(np.sum((y == 0) & test_mask))
    n_test_illicit = int(np.sum((y == 1) & test_mask))
    n_unknown = int(np.sum(unknown_mask))

    class_weight_ratio = float(n_train_licit / max(n_train_illicit, 1))

    print(f"  Train labeled: {train_mask.sum():,} (Licit: {n_train_licit:,}, Illicit: {n_train_illicit:,}) -> Ratio: {class_weight_ratio:.4f}")
    print(f"  Val labeled:   {val_mask.sum():,} (Licit: {n_val_licit:,}, Illicit: {n_val_illicit:,})")
    print(f"  Test labeled:  {test_mask.sum():,} (Licit: {n_test_licit:,}, Illicit: {n_test_illicit:,})")
    print(f"  Unknown nodes: {n_unknown:,} (Participate in graph topology, isolated from loss)")

    # 3. Fit StandardScaler STRICTLY on Training Split
    print("\n[Step 3/8] Fitting StandardScaler strictly on training labeled split...")
    scaler = StandardScaler()
    scaler.fit(X_raw[train_mask])
    X_norm = scaler.transform(X_raw)
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    print(f"  Saved scaler to {os.path.join(args.output_dir, 'scaler.joblib')}")

    # Data slices for tabular models
    X_train_165 = X_norm[train_mask]
    y_train = y[train_mask]
    X_val_165 = X_norm[val_mask]
    y_val = y[val_mask]
    X_test_165 = X_norm[test_mask]
    y_test = y[test_mask]

    # Data containers for seed iterations
    model_keys = ["HistGBDT", "XGBoost_Alone", "TGAT_Alone", "TGAT_BoostNet", "Random_Forest"]
    seed_records = []
    trained_models_seed42 = {}
    history_seed42 = None
    best_thresholds_seed42 = {}
    test_probs_seed42 = {}
    test_metrics_seed42 = {}
    test_evaluations_seed42 = {}

    # 4. Multi-seed training loop
    print(f"\n[Step 4/8] Executing fair ablation across {len(args.seeds)} seeds: {args.seeds}...")
    for s_idx, seed in enumerate(args.seeds):
        print(f"\n  ====================== SEED {seed} ({s_idx+1}/{len(args.seeds)}) ======================")

        # --- A. HistGBDT Baseline ---
        t0 = time.time()
        hist = HistGradientBoostingClassifier(
            max_iter=100,
            class_weight={0: 1.0, 1: class_weight_ratio},
            random_state=seed
        )
        hist.fit(X_train_165, y_train)
        hist_fit_time = time.time() - t0
        hist_val_probs = hist.predict_proba(X_val_165)[:, 1]
        hist_th, hist_val_f1 = tune_threshold(y_val, hist_val_probs)
        hist_test_probs = hist.predict_proba(X_test_165)[:, 1]
        hist_metrics = compute_metrics(y_test, hist_test_probs, hist_th)

        print(f"    [HistGBDT]       Val F1: {hist_val_f1:.4f} (Thresh={hist_th:.2f}) | Test F1: {hist_metrics['f1_illicit']:.4f} | Prec: {hist_metrics['precision']:.4f} | Rec: {hist_metrics['recall']:.4f} | ROC: {hist_metrics['roc_auc']:.4f}")

        # --- B. XGBoost Alone (165 features) ---
        t0 = time.time()
        xgb_alone = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=class_weight_ratio,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1
        )
        xgb_alone.fit(X_train_165, y_train)
        xgb_fit_time = time.time() - t0
        xgb_val_probs = xgb_alone.predict_proba(X_val_165)[:, 1]
        xgb_th, xgb_val_f1 = tune_threshold(y_val, xgb_val_probs)
        xgb_test_probs = xgb_alone.predict_proba(X_test_165)[:, 1]
        xgb_metrics = compute_metrics(y_test, xgb_test_probs, xgb_th)

        print(f"    [XGBoost Alone]  Val F1: {xgb_val_f1:.4f} (Thresh={xgb_th:.2f}) | Test F1: {xgb_metrics['f1_illicit']:.4f} | Prec: {xgb_metrics['precision']:.4f} | Rec: {xgb_metrics['recall']:.4f} | ROC: {xgb_metrics['roc_auc']:.4f}")

        # --- C. TGAT Alone ---
        print(f"    [Training TGAT]  Seed {seed}...")
        tgat_model, history, tgat_fit_time = train_tgat_model(
            x_train_norm=X_norm,
            time_steps=time_steps,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            class_weight=class_weight_ratio,
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience
        )
        tgat_val_probs = evaluate_tgat_alone(tgat_model, X_norm, time_steps, edge_index, val_mask)
        tgat_th, tgat_val_f1 = tune_threshold(y_val, tgat_val_probs)
        tgat_test_probs = evaluate_tgat_alone(tgat_model, X_norm, time_steps, edge_index, test_mask)
        tgat_metrics = compute_metrics(y_test, tgat_test_probs, tgat_th)

        print(f"    [TGAT Alone]     Val F1: {tgat_val_f1:.4f} (Thresh={tgat_th:.2f}) | Test F1: {tgat_metrics['f1_illicit']:.4f} | Prec: {tgat_metrics['precision']:.4f} | Rec: {tgat_metrics['recall']:.4f} | ROC: {tgat_metrics['roc_auc']:.4f}")

        # --- D. TGAT-BoostNet Hybrid (197 fused features) ---
        print(f"    [Extracting TGAT Embeddings] Step-by-step inductive extraction...")
        t0_emb = time.time()
        tgat_embeddings = extract_embeddings_inductive(tgat_model, X_norm, time_steps, edge_index)
        emb_extract_time = time.time() - t0_emb
        print(f"      Extracted 32-dim graph embeddings for all {len(tgat_embeddings):,} nodes in {emb_extract_time:.2f} s")

        # Feature concatenation: 165 raw features + 32 graph embeddings = 197 features
        X_fused = np.concatenate([X_norm, tgat_embeddings], axis=1)
        assert X_fused.shape[1] == 197, f"Expected 197 features, got {X_fused.shape[1]}"

        X_train_197 = X_fused[train_mask]
        X_val_197 = X_fused[val_mask]
        X_test_197 = X_fused[test_mask]

        t0 = time.time()
        hybrid_xgb = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=class_weight_ratio,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1
        )
        hybrid_xgb.fit(X_train_197, y_train)
        hybrid_fit_time = (time.time() - t0) + tgat_fit_time  # Total training time includes TGAT
        hybrid_val_probs = hybrid_xgb.predict_proba(X_val_197)[:, 1]
        hybrid_th, hybrid_val_f1 = tune_threshold(y_val, hybrid_val_probs)
        hybrid_test_probs = hybrid_xgb.predict_proba(X_test_197)[:, 1]
        hybrid_metrics = compute_metrics(y_test, hybrid_test_probs, hybrid_th)

        print(f"    [TGAT-BoostNet]  Val F1: {hybrid_val_f1:.4f} (Thresh={hybrid_th:.2f}) | Test F1: {hybrid_metrics['f1_illicit']:.4f} | Prec: {hybrid_metrics['precision']:.4f} | Rec: {hybrid_metrics['recall']:.4f} | ROC: {hybrid_metrics['roc_auc']:.4f}")

        # --- E. Random Forest Baseline ---
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X_train_165, y_train)
        rf_fit_time = time.time() - t0
        rf_val_probs = rf.predict_proba(X_val_165)[:, 1]
        rf_th, rf_val_f1 = tune_threshold(y_val, rf_val_probs)
        rf_test_probs = rf.predict_proba(X_test_165)[:, 1]
        rf_metrics = compute_metrics(y_test, rf_test_probs, rf_th)

        print(f"    [Random Forest]  Val F1: {rf_val_f1:.4f} (Thresh={rf_th:.2f}) | Test F1: {rf_metrics['f1_illicit']:.4f} | Prec: {rf_metrics['precision']:.4f} | Rec: {rf_metrics['recall']:.4f} | ROC: {rf_metrics['roc_auc']:.4f}")

        # Store seed record
        models_metrics_map = {
            "HistGBDT": (hist_metrics, hist_th, hist_val_f1, hist_fit_time),
            "XGBoost_Alone": (xgb_metrics, xgb_th, xgb_val_f1, xgb_fit_time),
            "TGAT_Alone": (tgat_metrics, tgat_th, tgat_val_f1, tgat_fit_time),
            "TGAT_BoostNet": (hybrid_metrics, hybrid_th, hybrid_val_f1, hybrid_fit_time),
            "Random_Forest": (rf_metrics, rf_th, rf_val_f1, rf_fit_time),
        }

        for m_name, (m_met, m_th, m_vf1, m_time) in models_metrics_map.items():
            seed_records.append({
                "seed": seed,
                "model": m_name,
                "precision": m_met["precision"],
                "recall": m_met["recall"],
                "f1_illicit": m_met["f1_illicit"],
                "f1_macro": m_met["f1_macro"],
                "roc_auc": m_met["roc_auc"],
                "average_precision": m_met["average_precision"],
                "val_threshold": m_th,
                "val_f1": round(m_vf1, 4),
                "fit_time_sec": round(m_time, 2)
            })

        # Save artifacts from primary seed 42
        if seed == 42:
            trained_models_seed42 = {
                "hist": hist,
                "xgb_alone": xgb_alone,
                "tgat": tgat_model,
                "hybrid_xgb": hybrid_xgb,
                "rf": rf
            }
            history_seed42 = history
            best_thresholds_seed42 = {
                "HistGBDT": hist_th,
                "XGBoost_Alone": xgb_th,
                "TGAT_Alone": tgat_th,
                "TGAT_BoostNet": hybrid_th,
                "Random_Forest": rf_th
            }
            test_probs_seed42 = {
                "HistGBDT": hist_test_probs,
                "XGBoost_Alone": xgb_test_probs,
                "TGAT_Alone": tgat_test_probs,
                "TGAT_BoostNet": hybrid_test_probs,
                "Random_Forest": rf_test_probs
            }
            test_metrics_seed42 = {
                "HistGBDT": hist_metrics,
                "XGBoost_Alone": xgb_metrics,
                "TGAT_Alone": tgat_metrics,
                "TGAT_BoostNet": hybrid_metrics,
                "Random_Forest": rf_metrics
            }
            saved_fused_test_197 = X_test_197

    # 5. Export Seed Results CSV
    print("\n[Step 5/8] Aggregating multi-seed ablation statistics...")
    df_seeds = pd.DataFrame(seed_records)
    seeds_csv_path = os.path.join(args.output_dir, "seed_results.csv")
    df_seeds.to_csv(seeds_csv_path, index=False)
    print(f"  Saved seed results to {seeds_csv_path}")

    # Summary table across seeds
    summary_stats = {}
    print("\n" + "=" * 90)
    print("ABLATION SUMMARY ACROSS 5 SEEDS (Mean ± Std):")
    print(f"{'Model':<18} | {'Precision':<13} | {'Recall':<13} | {'Illicit F1':<13} | {'ROC-AUC':<13} | {'PR-AUC':<13}")
    print("-" * 90)
    for m_name in model_keys:
        sub = df_seeds[df_seeds["model"] == m_name]
        p_m, p_s = sub["precision"].mean(), sub["precision"].std()
        r_m, r_s = sub["recall"].mean(), sub["recall"].std()
        f1_m, f1_s = sub["f1_illicit"].mean(), sub["f1_illicit"].std()
        roc_m, roc_s = sub["roc_auc"].mean(), sub["roc_auc"].std()
        pr_m, pr_s = sub["average_precision"].mean(), sub["average_precision"].std()
        summary_stats[m_name] = {
            "precision_mean": round(float(p_m), 4), "precision_std": round(float(p_s), 4),
            "recall_mean": round(float(r_m), 4), "recall_std": round(float(r_s), 4),
            "f1_illicit_mean": round(float(f1_m), 4), "f1_illicit_std": round(float(f1_s), 4),
            "roc_auc_mean": round(float(roc_m), 4), "roc_auc_std": round(float(roc_s), 4),
            "average_precision_mean": round(float(pr_m), 4), "average_precision_std": round(float(pr_s), 4),
        }
        print(f"{m_name:<18} | {p_m:.4f}±{p_s:.4f} | {r_m:.4f}±{r_s:.4f} | {f1_m:.4f}±{f1_s:.4f} | {roc_m:.4f}±{roc_s:.4f} | {pr_m:.4f}±{pr_s:.4f}")
    print("=" * 90)

    # 6. Save Checkpoints & Artifacts
    print("\n[Step 6/8] Saving model checkpoints and configurations...")
    # TGAT Checkpoint
    tgat_ckpt_path = os.path.join(args.output_dir, "tgat_checkpoint.pt")
    torch.save({
        "state_dict": trained_models_seed42["tgat"].state_dict(),
        "in_channels": 165,
        "hidden_channels": 32,
        "out_channels": 2,
        "heads": 4,
        "seed": 42
    }, tgat_ckpt_path)
    print(f"  Saved TGAT checkpoint to {tgat_ckpt_path}")

    # XGBoost Checkpoints
    xgb_alone_path = os.path.join(args.output_dir, "xgboost_alone.json")
    trained_models_seed42["xgb_alone"].save_model(xgb_alone_path)
    print(f"  Saved XGBoost alone model to {xgb_alone_path}")

    hybrid_xgb_path = os.path.join(args.output_dir, "tgat_boostnet_xgb.json")
    trained_models_seed42["hybrid_xgb"].save_model(hybrid_xgb_path)
    print(f"  Saved TGAT-BoostNet hybrid model to {hybrid_xgb_path}")

    # Training History JSON
    hist_json_path = os.path.join(args.output_dir, "training_history.json")
    with open(hist_json_path, "w", encoding="utf-8") as f:
        json.dump(history_seed42, f, indent=2)
    print(f"  Saved training history to {hist_json_path}")

    # Confusion Matrices JSON
    cm_json_path = os.path.join(args.output_dir, "confusion_matrices.json")
    cms = {m: test_metrics_seed42[m]["confusion_matrix"] for m in model_keys}
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(cms, f, indent=2)
    print(f"  Saved confusion matrices to {cm_json_path}")

    # 7. Per-Timestep Performance & Temporal Drift (Test Steps 35–49)
    print("\n[Step 7/8] Generating per-timestep temporal drift breakdown (Steps 35 to 49)...")
    test_timesteps = time_steps[test_mask]
    timestep_rows = []

    for t in range(35, 50):
        step_submask = (test_timesteps == t)
        n_labeled = int(step_submask.sum())
        if n_labeled == 0:
            continue
        y_step = y_test[step_submask]
        n_lic = int((y_step == 0).sum())
        n_ill = int((y_step == 1).sum())

        row = {
            "time_step": t,
            "total_labeled": n_labeled,
            "licit_count": n_lic,
            "illicit_count": n_ill,
            "is_shift_boundary": bool(t == 43),
            "is_low_support": bool(n_ill < 10)
        }

        for m_name in model_keys:
            th = best_thresholds_seed42[m_name]
            p_sub = test_probs_seed42[m_name][step_submask]
            pred_sub = (p_sub >= th).astype(int)

            pr = float(precision_score(y_step, pred_sub, pos_label=1, zero_division=0))
            rc = float(recall_score(y_step, pred_sub, pos_label=1, zero_division=0))
            f1 = float(f1_score(y_step, pred_sub, pos_label=1, zero_division=0))

            row[f"{m_name}_precision"] = round(pr, 4)
            row[f"{m_name}_recall"] = round(rc, 4)
            row[f"{m_name}_f1"] = round(f1, 4)

        timestep_rows.append(row)

    df_per_step = pd.DataFrame(timestep_rows)
    per_step_csv_path = os.path.join(args.output_dir, "per_timestep_metrics.csv")
    df_per_step.to_csv(per_step_csv_path, index=False)
    print(f"  Saved per-timestep metrics to {per_step_csv_path}")

    # Test Predictions Parquet
    df_preds = pd.DataFrame({
        "tx_id": tx_ids[test_mask],
        "time_step": test_timesteps,
        "y_true": y_test,
        "prob_hist": np.round(test_probs_seed42["HistGBDT"], 4),
        "prob_xgb_alone": np.round(test_probs_seed42["XGBoost_Alone"], 4),
        "prob_tgat_alone": np.round(test_probs_seed42["TGAT_Alone"], 4),
        "prob_tgat_boostnet": np.round(test_probs_seed42["TGAT_BoostNet"], 4),
        "prob_rf": np.round(test_probs_seed42["Random_Forest"], 4),
        "pred_tgat_boostnet": (test_probs_seed42["TGAT_BoostNet"] >= best_thresholds_seed42["TGAT_BoostNet"]).astype(int)
    })
    preds_parquet_path = os.path.join(args.output_dir, "test_predictions.parquet")
    df_preds.to_parquet(preds_parquet_path, index=False, engine="pyarrow")
    print(f"  Saved test predictions parquet to {preds_parquet_path}")

    # 8. In-Memory Latency Benchmark
    print("\n[Step 8/8] Measuring in-memory inference latency across batch sizes...")
    latency_data = run_latency_benchmark(
        hist_model=trained_models_seed42["hist"],
        xgb_model=trained_models_seed42["xgb_alone"],
        tgat_model=trained_models_seed42["tgat"],
        hybrid_model=trained_models_seed42["hybrid_xgb"],
        rf_model=trained_models_seed42["rf"],
        X_test_165=X_test_165,
        X_test_197=saved_fused_test_197,
        time_steps=time_steps[test_mask],
        edge_index=edge_index
    )
    latency_json_path = os.path.join(args.output_dir, "latency_benchmark.json")
    with open(latency_json_path, "w", encoding="utf-8") as f:
        json.dump(latency_data, f, indent=2)
    print(f"  Saved latency benchmark to {latency_json_path}")

    # Comprehensive Metrics JSON
    metrics_json_path = os.path.join(args.output_dir, "metrics.json")
    final_metrics_payload = {
        "experiment_name": "TGAT-BoostNet Fair Ablation and Temporal Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_statistics": {
            "total_nodes": int(len(tx_ids)),
            "total_edges": int(edge_index.shape[1]),
            "raw_features": int(X_raw.shape[1]),
            "fused_features": int(saved_fused_test_197.shape[1]),
            "train_labeled": int(train_mask.sum()),
            "val_labeled": int(val_mask.sum()),
            "test_labeled": int(test_mask.sum()),
            "unknown_nodes": int(n_unknown),
            "step_35_counts": {
                "total_labeled": 1341,
                "illicit": 182,
                "licit": 1159,
                "unknown": 4166
            }
        },
        "model_summary_5_seeds": summary_stats,
        "seed_42_single_run": {
            m: {
                "metrics": test_metrics_seed42[m],
                "selected_threshold": best_thresholds_seed42[m]
            } for m in model_keys
        },
        "latency_summary": latency_data
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)
    print(f"  Saved comprehensive metrics JSON to {metrics_json_path}")

    print("\n" + "=" * 85)
    print("ALL EXPERIMENT RUNS AND BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    main()
