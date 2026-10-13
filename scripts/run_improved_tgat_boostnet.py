"""
Production Runner for Improved TGAT-BoostNet Comparative Ablation and Benchmarking.

Evaluates the 7 required configurations across 5 fixed random seeds [42, 123, 456, 789, 2024]
under a strict chronological protocol:
- Train: Timesteps 1 to 29
- Val: Timesteps 30 to 34 (Threshold tuning & checkpoint selection)
- Test: Timesteps 35 to 49 (Single test evaluation at validation thresholds)

Configurations:
1. HistGBDT Baseline (165 Tabular Features)
2. XGBoost Baseline (165 Tabular Features)
3. SpatioTemporalGAT Alone (32-dim Graph Representation)
4. TGAT-BoostNet Baseline (197 Fused Features: 165 Tabular + 32 TGAT Embeddings)
5. TGAT-HistGBDT (197 Fused Features: 165 Tabular + 32 TGAT Embeddings)
6. Tuned XGBoost (165 Tabular Features without Graph)
7. TGAT-BoostNet Optimized (197 Fused Features with Regularized Booster)
+ Hybrid Ensemble (TGAT-BoostNet + HistGBDT)

Artifacts written to: data/validation/tgat_boostnet_improved/
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.gat import SpatioTemporalGAT, is_gat_available

SEEDS = [42, 123, 456, 789, 2024]
DATA_PATH = "data/processed/elliptic/elliptic_preprocessed.npz"
OUTPUT_DIR = "data/validation/tgat_boostnet_improved"


def parse_args():
    parser = argparse.ArgumentParser(description="Improved TGAT-BoostNet 7-Configuration Ablation Runner")
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def load_dataset(data_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")
    data = np.load(data_path)
    return (
        data["tx_ids"],
        data["time_steps"],
        data["X_raw"],
        data["y"],
        data["edge_index"],
        data["edge_time_steps"]
    )


def tune_val_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> Tuple[float, float]:
    best_th, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        p = (val_probs >= t).astype(int)
        score = f1_score(y_val, p, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = float(t)
    return float(best_th), float(best_f1)


def eval_model(y_true: np.ndarray, test_probs: np.ndarray, thresh: float) -> Dict[str, Any]:
    p = (test_probs >= thresh).astype(int)
    cm = confusion_matrix(y_true, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = float(precision_score(y_true, p, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, p, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, p, pos_label=1, zero_division=0))
    f1_m = float(f1_score(y_true, p, average="macro", zero_division=0))
    roc = float(roc_auc_score(y_true, test_probs))
    ap = float(average_precision_score(y_true, test_probs))
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_illicit": round(f1, 4),
        "f1_macro": round(f1_m, 4),
        "roc_auc": round(roc, 4),
        "average_precision": round(ap, 4),
        "threshold": round(float(thresh), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }


def train_tgat_seed(
    x_norm: np.ndarray,
    time_steps: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    class_weight: float,
    seed: int,
    epochs: int = 30,
    lr: float = 0.005,
    patience: int = 10,
    device: str = "cpu"
) -> Tuple[SpatioTemporalGAT, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    src, dst = edge_index[0], edge_index[1]
    train_edge_mask = (time_steps[src] <= 29) & (time_steps[dst] <= 29)
    train_edge_index = torch.tensor(edge_index[:, train_edge_mask], dtype=torch.long, device=device)
    full_edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)

    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(time_steps.reshape(-1, 1) / 49.0, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool, device=device)

    model = SpatioTemporalGAT(in_channels=165, hidden_channels=32, out_channels=2, heads=4, dropout=0.3).to(device)
    weights = torch.tensor([1.0, class_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_v_f1 = -1.0
    best_weights = None
    best_epoch = 0
    patience_cnt = 0
    history = {"epochs": [], "train_loss": [], "val_f1": []}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor, train_edge_index, t_tensor)
        loss = criterion(out[train_mask_t], y_tensor[train_mask_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(x_tensor, full_edge_index, t_tensor)
            v_probs = F.softmax(v_logits[val_mask_t], dim=1)[:, 1].cpu().numpy()
            _, vf1 = tune_val_threshold(y[val_mask], v_probs)

        history["epochs"].append(ep)
        history["train_loss"].append(round(float(loss.item()), 4))
        history["val_f1"].append(round(float(vf1), 4))

        if vf1 > best_v_f1:
            best_v_f1 = vf1
            best_epoch = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    train_time = time.time() - t0
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    history["best_epoch"] = best_epoch
    history["best_val_f1"] = round(float(best_v_f1), 4)

    # Inductive embedding extraction
    embeddings = np.zeros((len(x_norm), 32), dtype=np.float32)
    with torch.no_grad():
        for t in range(1, 50):
            step_mask = (time_steps == t)
            step_indices = np.where(step_mask)[0]
            if len(step_indices) == 0:
                continue
            sub_edge_mask = (time_steps[src] == t) & (time_steps[dst] == t)
            sub_edge_index = edge_index[:, sub_edge_mask]
            idx_map = {g: l for l, g in enumerate(step_indices)}
            if sub_edge_index.shape[1] > 0:
                l_src = [idx_map[s] for s in sub_edge_index[0] if s in idx_map]
                l_dst = [idx_map[d] for d in sub_edge_index[1] if d in idx_map]
                local_e = torch.tensor([l_src, l_dst], dtype=torch.long, device=device)
            else:
                local_e = torch.empty((2, 0), dtype=torch.long, device=device)
            xt = torch.tensor(x_norm[step_indices], dtype=torch.float32, device=device)
            tt = torch.tensor(time_steps[step_indices].reshape(-1, 1) / 49.0, dtype=torch.float32, device=device)
            embeddings[step_indices] = model.extract_embedding(xt, local_e, tt).cpu().numpy()

    with torch.no_grad():
        tgat_val_probs = F.softmax(model(x_tensor, full_edge_index, t_tensor)[val_mask_t], dim=1)[:, 1].cpu().numpy()
        tgat_test_probs = F.softmax(model(x_tensor, full_edge_index, t_tensor)[torch.tensor(y != -1, dtype=torch.bool) & (torch.tensor(time_steps >= 35, dtype=torch.bool))], dim=1)[:, 1].cpu().numpy()

    return model, embeddings, tgat_val_probs, tgat_test_probs, history, train_time


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 90)
    print("IMPROVED TGAT-BOOSTNET: CONTROLLED 7-CONFIGURATION ABLATION & TEMPORAL AUDIT")
    print(f"Data source:      {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seeds:     {args.seeds}")
    print("=" * 90)

    # 1. Load Data
    print("\n[Step 1/8] Loading Elliptic dataset...")
    tx_ids, time_steps, X_raw, y, edge_index, edge_time_steps = load_dataset(args.data_path)
    print(f"  Loaded {len(tx_ids):,} nodes across {time_steps.max()} timesteps.")
    print(f"  Raw features: {X_raw.shape[1]}, Directed edges: {edge_index.shape[1]:,}")

    # 2. Strict Partition Masks
    labeled = (y != -1)
    train_mask = (time_steps <= 29) & labeled
    val_mask = (time_steps >= 30) & (time_steps <= 34) & labeled
    test_mask = (time_steps >= 35) & labeled
    unknown_mask = (y == -1)

    n_train_licit = int(np.sum((y == 0) & train_mask))
    n_train_illicit = int(np.sum((y == 1) & train_mask))
    class_weight_ratio = float(n_train_licit / max(n_train_illicit, 1))

    print(f"  Train: {train_mask.sum():,} nodes | Val: {val_mask.sum():,} nodes | Test: {test_mask.sum():,} nodes")
    print(f"  Unknown nodes: {unknown_mask.sum():,} (Isolated from loss/eval, participate in graph)")
    print(f"  Train imbalance ratio: {class_weight_ratio:.4f} : 1")

    # 3. Fit StandardScaler strictly on training labeled split
    scaler = StandardScaler()
    scaler.fit(X_raw[train_mask])
    X_norm = scaler.transform(X_raw)
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))

    X_train_165 = X_norm[train_mask]
    y_train = y[train_mask]
    X_val_165 = X_norm[val_mask]
    y_val = y[val_mask]
    X_test_165 = X_norm[test_mask]
    y_test = y[test_mask]

    # Containers
    seed_records = []
    models_seed42 = {}
    history_seed42 = None
    best_thresholds_seed42 = {}
    test_probs_seed42 = {}
    test_metrics_seed42 = {}

    configs = [
        "1_HistGBDT_165",
        "2_XGBoost_Baseline_165",
        "3_TGAT_Alone_32",
        "4_TGAT_BoostNet_Baseline_197",
        "5_TGAT_HistGBDT_197",
        "6_Tuned_XGBoost_165",
        "7_TGAT_BoostNet_Optimized_197",
        "8_TGAT_Hybrid_Ensemble"
    ]

    print(f"\n[Step 2/8] Executing multi-seed evaluation across {len(args.seeds)} seeds...")
    for s_idx, seed in enumerate(args.seeds):
        print(f"\n  ====================== SEED {seed} ({s_idx+1}/{len(args.seeds)}) ======================")

        # Train TGAT and extract 32-dim graph embeddings
        tgat_model, emb, tgat_val_p, tgat_test_p, hist_tgat, tgat_train_time = train_tgat_seed(
            x_norm=X_norm,
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

        X_fused_197 = np.concatenate([X_norm, emb], axis=1)
        X_train_197 = X_fused_197[train_mask]
        X_val_197 = X_fused_197[val_mask]
        X_test_197 = X_fused_197[test_mask]

        # Config 1: HistGBDT (165)
        t0 = time.time()
        hgb_165 = HistGradientBoostingClassifier(max_iter=100, class_weight={0: 1.0, 1: class_weight_ratio}, random_state=seed)
        hgb_165.fit(X_train_165, y_train)
        time_1 = time.time() - t0
        val_p1 = hgb_165.predict_proba(X_val_165)[:, 1]
        th1, vf1_1 = tune_val_threshold(y_val, val_p1)
        test_p1 = hgb_165.predict_proba(X_test_165)[:, 1]
        res1 = eval_model(y_test, test_p1, th1)

        # Config 2: XGBoost Baseline (165)
        t0 = time.time()
        xgb_165 = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=class_weight_ratio, random_state=seed, eval_metric="logloss", n_jobs=-1)
        xgb_165.fit(X_train_165, y_train)
        time_2 = time.time() - t0
        val_p2 = xgb_165.predict_proba(X_val_165)[:, 1]
        th2, vf1_2 = tune_val_threshold(y_val, val_p2)
        test_p2 = xgb_165.predict_proba(X_test_165)[:, 1]
        res2 = eval_model(y_test, test_p2, th2)

        # Config 3: TGAT Alone (32)
        th3, vf1_3 = tune_val_threshold(y_val, tgat_val_p)
        res3 = eval_model(y_test, tgat_test_p, th3)
        time_3 = tgat_train_time

        # Config 4: TGAT-BoostNet Baseline (197)
        t0 = time.time()
        xgb_197 = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=class_weight_ratio, random_state=seed, eval_metric="logloss", n_jobs=-1)
        xgb_197.fit(X_train_197, y_train)
        time_4 = (time.time() - t0) + tgat_train_time
        val_p4 = xgb_197.predict_proba(X_val_197)[:, 1]
        th4, vf1_4 = tune_val_threshold(y_val, val_p4)
        test_p4 = xgb_197.predict_proba(X_test_197)[:, 1]
        res4 = eval_model(y_test, test_p4, th4)

        # Config 5: TGAT-HistGBDT (197)
        t0 = time.time()
        hgb_197 = HistGradientBoostingClassifier(max_iter=100, class_weight={0: 1.0, 1: class_weight_ratio}, random_state=seed)
        hgb_197.fit(X_train_197, y_train)
        time_5 = (time.time() - t0) + tgat_train_time
        val_p5 = hgb_197.predict_proba(X_val_197)[:, 1]
        th5, vf1_5 = tune_val_threshold(y_val, val_p5)
        test_p5 = hgb_197.predict_proba(X_test_197)[:, 1]
        res5 = eval_model(y_test, test_p5, th5)

        # Config 6: Tuned XGBoost (165 without graph)
        t0 = time.time()
        xgb_tuned_165 = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0, scale_pos_weight=class_weight_ratio, random_state=seed, eval_metric="logloss", n_jobs=-1)
        xgb_tuned_165.fit(X_train_165, y_train)
        time_6 = time.time() - t0
        val_p6 = xgb_tuned_165.predict_proba(X_val_165)[:, 1]
        th6, vf1_6 = tune_val_threshold(y_val, val_p6)
        test_p6 = xgb_tuned_165.predict_proba(X_test_165)[:, 1]
        res6 = eval_model(y_test, test_p6, th6)

        # Config 7: TGAT-BoostNet Optimized (197 with regularization)
        t0 = time.time()
        xgb_opt_197 = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0, scale_pos_weight=class_weight_ratio, random_state=seed, eval_metric="logloss", n_jobs=-1)
        xgb_opt_197.fit(X_train_197, y_train)
        time_7 = (time.time() - t0) + tgat_train_time
        val_p7 = xgb_opt_197.predict_proba(X_val_197)[:, 1]
        th7, vf1_7 = tune_val_threshold(y_val, val_p7)
        test_p7 = xgb_opt_197.predict_proba(X_test_197)[:, 1]
        res7 = eval_model(y_test, test_p7, th7)

        # Config 8: TGAT Hybrid Ensemble (0.5 TGAT-BoostNet + 0.5 HistGBDT)
        val_p8 = 0.5 * val_p4 + 0.5 * val_p1
        th8, vf1_8 = tune_val_threshold(y_val, val_p8)
        test_p8 = 0.5 * test_p4 + 0.5 * test_p1
        res8 = eval_model(y_test, test_p8, th8)
        time_8 = time_4 + time_1

        current_run_map = {
            "1_HistGBDT_165": (res1, th1, vf1_1, time_1),
            "2_XGBoost_Baseline_165": (res2, th2, vf1_2, time_2),
            "3_TGAT_Alone_32": (res3, th3, vf1_3, time_3),
            "4_TGAT_BoostNet_Baseline_197": (res4, th4, vf1_4, time_4),
            "5_TGAT_HistGBDT_197": (res5, th5, vf1_5, time_5),
            "6_Tuned_XGBoost_165": (res6, th6, vf1_6, time_6),
            "7_TGAT_BoostNet_Optimized_197": (res7, th7, vf1_7, time_7),
            "8_TGAT_Hybrid_Ensemble": (res8, th8, vf1_8, time_8)
        }

        for c_name, (c_res, c_th, c_vf1, c_time) in current_run_map.items():
            seed_records.append({
                "seed": seed,
                "config": c_name,
                "precision": c_res["precision"],
                "recall": c_res["recall"],
                "f1_illicit": c_res["f1_illicit"],
                "f1_macro": c_res["f1_macro"],
                "roc_auc": c_res["roc_auc"],
                "average_precision": c_res["average_precision"],
                "val_threshold": c_th,
                "val_f1": round(c_vf1, 4),
                "fit_time_sec": round(c_time, 2)
            })

        print(f"    HistGBDT (165):            F1: {res1['f1_illicit']:.4f} | Prec: {res1['precision']:.4f} | Rec: {res1['recall']:.4f} | ROC: {res1['roc_auc']:.4f}")
        print(f"    TGAT-BoostNet Base (197):  F1: {res4['f1_illicit']:.4f} | Prec: {res4['precision']:.4f} | Rec: {res4['recall']:.4f} | ROC: {res4['roc_auc']:.4f}")
        print(f"    TGAT-BoostNet Opt (197):   F1: {res7['f1_illicit']:.4f} | Prec: {res7['precision']:.4f} | Rec: {res7['recall']:.4f} | ROC: {res7['roc_auc']:.4f}")
        print(f"    TGAT Hybrid Ensemble:      F1: {res8['f1_illicit']:.4f} | Prec: {res8['precision']:.4f} | Rec: {res8['recall']:.4f} | ROC: {res8['roc_auc']:.4f}")

        if seed == 42:
            models_seed42 = {
                "tgat": tgat_model,
                "hgb_165": hgb_165,
                "xgb_165": xgb_165,
                "xgb_197": xgb_197,
                "hgb_197": hgb_197,
                "xgb_opt_197": xgb_opt_197
            }
            history_seed42 = hist_tgat
            best_thresholds_seed42 = {c: current_run_map[c][1] for c in configs}
            test_probs_seed42 = {
                "1_HistGBDT_165": test_p1,
                "2_XGBoost_Baseline_165": test_p2,
                "3_TGAT_Alone_32": tgat_test_p,
                "4_TGAT_BoostNet_Baseline_197": test_p4,
                "5_TGAT_HistGBDT_197": test_p5,
                "6_Tuned_XGBoost_165": test_p6,
                "7_TGAT_BoostNet_Optimized_197": test_p7,
                "8_TGAT_Hybrid_Ensemble": test_p8
            }
            test_metrics_seed42 = {c: current_run_map[c][0] for c in configs}
            saved_fused_197_seed42 = X_test_197

    # 4. Save seed results CSV
    df_seeds = pd.DataFrame(seed_records)
    seeds_csv_path = os.path.join(args.output_dir, "seed_results.csv")
    df_seeds.to_csv(seeds_csv_path, index=False)
    print(f"\n[Step 3/8] Saved seed results to {seeds_csv_path}")

    # Summary table across seeds
    summary_stats = {}
    print("\n" + "=" * 105)
    print("SUMMARY COMPARISON ACROSS 5 SEEDS (Mean ± Std):")
    print(f"{'Configuration':<30} | {'Precision':<13} | {'Recall':<13} | {'Illicit F1':<13} | {'ROC-AUC':<13} | {'PR-AUC':<13}")
    print("-" * 105)
    for c_name in configs:
        sub = df_seeds[df_seeds["config"] == c_name]
        pm, ps = sub["precision"].mean(), sub["precision"].std()
        rm, rs = sub["recall"].mean(), sub["recall"].std()
        fm, fs = sub["f1_illicit"].mean(), sub["f1_illicit"].std()
        rocm, rocs = sub["roc_auc"].mean(), sub["roc_auc"].std()
        apm, aps = sub["average_precision"].mean(), sub["average_precision"].std()
        summary_stats[c_name] = {
            "precision_mean": round(float(pm), 4), "precision_std": round(float(ps), 4),
            "recall_mean": round(float(rm), 4), "recall_std": round(float(rs), 4),
            "f1_illicit_mean": round(float(fm), 4), "f1_illicit_std": round(float(fs), 4),
            "roc_auc_mean": round(float(rocm), 4), "roc_auc_std": round(float(rocs), 4),
            "average_precision_mean": round(float(apm), 4), "average_precision_std": round(float(aps), 4)
        }
        print(f"{c_name:<30} | {pm:.4f}±{ps:.4f} | {rm:.4f}±{rs:.4f} | {fm:.4f}±{fs:.4f} | {rocm:.4f}±{rocs:.4f} | {apm:.4f}±{aps:.4f}")
    print("=" * 105)

    # 5. Save Model Checkpoints
    print("\n[Step 4/8] Saving trained model checkpoints...")
    torch.save({
        "state_dict": models_seed42["tgat"].state_dict(),
        "in_channels": 165,
        "hidden_channels": 32,
        "out_channels": 2,
        "heads": 4
    }, os.path.join(args.output_dir, "tgat_checkpoint.pt"))

    models_seed42["xgb_197"].save_model(os.path.join(args.output_dir, "tgat_boostnet_xgb.json"))
    models_seed42["xgb_opt_197"].save_model(os.path.join(args.output_dir, "tgat_boostnet_opt_xgb.json"))
    models_seed42["xgb_165"].save_model(os.path.join(args.output_dir, "xgboost_baseline.json"))
    joblib.dump(models_seed42["hgb_165"], os.path.join(args.output_dir, "histgbdt_baseline.joblib"))

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history_seed42, f, indent=2)

    cms = {c: test_metrics_seed42[c]["confusion_matrix"] for c in configs}
    with open(os.path.join(args.output_dir, "confusion_matrices.json"), "w") as f:
        json.dump(cms, f, indent=2)

    # 6. Per-Timestep Performance & Drift
    print("\n[Step 5/8] Computing per-timestep drift metrics (Steps 35 to 49)...")
    test_timesteps = time_steps[test_mask]
    step_rows = []
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

        for c_name in configs:
            th = best_thresholds_seed42[c_name]
            p_sub = test_probs_seed42[c_name][step_submask]
            pred_sub = (p_sub >= th).astype(int)
            pr = float(precision_score(y_step, pred_sub, pos_label=1, zero_division=0))
            rc = float(recall_score(y_step, pred_sub, pos_label=1, zero_division=0))
            f1 = float(f1_score(y_step, pred_sub, pos_label=1, zero_division=0))
            row[f"{c_name}_precision"] = round(pr, 4)
            row[f"{c_name}_recall"] = round(rc, 4)
            row[f"{c_name}_f1"] = round(f1, 4)

        step_rows.append(row)

    df_step = pd.DataFrame(step_rows)
    df_step.to_csv(os.path.join(args.output_dir, "per_timestep_metrics.csv"), index=False)

    # 7. Test Predictions Parquet
    print("\n[Step 6/8] Exporting test predictions parquet...")
    df_preds = pd.DataFrame({
        "tx_id": tx_ids[test_mask],
        "time_step": test_timesteps,
        "y_true": y_test,
        "prob_hist_165": np.round(test_probs_seed42["1_HistGBDT_165"], 4),
        "prob_xgb_165": np.round(test_probs_seed42["2_XGBoost_Baseline_165"], 4),
        "prob_tgat_alone": np.round(test_probs_seed42["3_TGAT_Alone_32"], 4),
        "prob_tgat_boostnet_base": np.round(test_probs_seed42["4_TGAT_BoostNet_Baseline_197"], 4),
        "prob_tgat_histgbdt": np.round(test_probs_seed42["5_TGAT_HistGBDT_197"], 4),
        "prob_tgat_boostnet_opt": np.round(test_probs_seed42["7_TGAT_BoostNet_Optimized_197"], 4),
        "prob_hybrid_ensemble": np.round(test_probs_seed42["8_TGAT_Hybrid_Ensemble"], 4),
        "pred_tgat_boostnet_base": (test_probs_seed42["4_TGAT_BoostNet_Baseline_197"] >= best_thresholds_seed42["4_TGAT_BoostNet_Baseline_197"]).astype(int),
        "pred_tgat_boostnet_opt": (test_probs_seed42["7_TGAT_BoostNet_Optimized_197"] >= best_thresholds_seed42["7_TGAT_BoostNet_Optimized_197"]).astype(int),
        "pred_hybrid_ensemble": (test_probs_seed42["8_TGAT_Hybrid_Ensemble"] >= best_thresholds_seed42["8_TGAT_Hybrid_Ensemble"]).astype(int)
    })
    df_preds.to_parquet(os.path.join(args.output_dir, "test_predictions.parquet"), index=False, engine="pyarrow")

    # 8. In-Memory Latency Benchmark
    print("\n[Step 7/8] Measuring in-memory inference latency across batch sizes...")
    batch_sizes = [1, 64, 1000, len(X_test_165)]
    latency_dict = {}

    for b in batch_sizes:
        b_key = str(b)
        latency_dict[b_key] = {}
        s165 = X_test_165[:b]
        s197 = saved_fused_197_seed42[:b]

        for m_label, model_obj, data_sub in [
            ("HistGBDT_165", models_seed42["hgb_165"], s165),
            ("XGBoost_165", models_seed42["xgb_165"], s165),
            ("TGAT_BoostNet_Base_197", models_seed42["xgb_197"], s197),
            ("TGAT_BoostNet_Opt_197", models_seed42["xgb_opt_197"], s197)
        ]:
            for _ in range(5):
                _ = model_obj.predict_proba(data_sub)
            times = []
            n_iters = 50 if b <= 64 else 15
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _ = model_obj.predict_proba(data_sub)
                times.append((time.perf_counter() - t0) * 1000.0)
            latency_dict[b_key][m_label] = {
                "median_ms": round(float(np.median(times)), 4),
                "p95_ms": round(float(np.percentile(times, 95)), 4),
                "per_tx_us": round(float(np.median(times) / b * 1000.0), 2)
            }

    with open(os.path.join(args.output_dir, "latency_benchmark.json"), "w") as f:
        json.dump(latency_dict, f, indent=2)

    # 9. Final Metrics JSON
    final_payload = {
        "experiment_name": "Improved TGAT-BoostNet 7-Configuration Ablation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_statistics": {
            "total_nodes": len(tx_ids),
            "total_edges": edge_index.shape[1],
            "raw_features": 165,
            "fused_features": 197,
            "train_labeled": int(train_mask.sum()),
            "val_labeled": int(val_mask.sum()),
            "test_labeled": int(test_mask.sum()),
            "unknown_nodes": int(unknown_mask.sum())
        },
        "model_summary_5_seeds": summary_stats,
        "seed_42_single_run": {
            c: {
                "metrics": test_metrics_seed42[c],
                "selected_threshold": best_thresholds_seed42[c]
            } for c in configs
        },
        "latency_summary": latency_dict
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n[Step 8/8] Experiment completed successfully. Metrics written to {os.path.join(args.output_dir, 'metrics.json')}")
    print("=" * 90)


if __name__ == "__main__":
    main()
