"""
Leakage-safe training/evaluation utilities for the Elliptic supervised benchmark.

Protocol (fixed): train = labelled steps 1-29, validation = labelled steps 30-34,
test = labelled steps 35-49. Unknown-label nodes (y == -1) stay in the graph for
message passing but never enter a loss, a threshold search or a metric.
Hyper-parameters, thresholds and the final model choice use validation only.
"""
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

TRAIN_STEPS = (1, 29)
VAL_STEPS = (30, 34)
TEST_STEPS = (35, 49)
THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 2)


@dataclass
class EllipticData:
    tx_ids: np.ndarray
    time_steps: np.ndarray
    X_raw: np.ndarray
    y: np.ndarray
    edge_index: np.ndarray

    @property
    def labelled(self):
        return self.y != -1

    def split_mask(self, steps: Tuple[int, int]) -> np.ndarray:
        return (self.time_steps >= steps[0]) & (self.time_steps <= steps[1]) & self.labelled

    @property
    def train_mask(self):
        return self.split_mask(TRAIN_STEPS)

    @property
    def val_mask(self):
        return self.split_mask(VAL_STEPS)

    @property
    def test_mask(self):
        return self.split_mask(TEST_STEPS)


def load_npz(path: str) -> EllipticData:
    d = np.load(path)
    return EllipticData(d["tx_ids"], d["time_steps"].astype(int), d["X_raw"].astype(np.float32),
                        d["y"].astype(int), d["edge_index"].astype(np.int64))


def load_spark_output(directory: str) -> EllipticData:
    """Load the Spark-produced Elliptic dataset (see scripts/process_elliptic_spark.py)."""
    import os
    import pandas as pd
    nodes = pd.read_parquet(os.path.join(directory, "nodes"))
    nodes = nodes.sort_values("node_idx").reset_index(drop=True)
    edges = pd.read_parquet(os.path.join(directory, "edges")).sort_values("edge_id")
    feat_cols = [f"f{i}" for i in range(1, 166)]
    return EllipticData(nodes["tx_id"].to_numpy(np.int64), nodes["time_step"].to_numpy(int),
                        nodes[feat_cols].to_numpy(np.float32), nodes["label"].to_numpy(int),
                        np.vstack([edges["source_idx"].to_numpy(np.int64), edges["target_idx"].to_numpy(np.int64)]))


def assert_no_cross_step_edges(data: EllipticData) -> None:
    s, t = data.edge_index
    n_cross = int((data.time_steps[s] != data.time_steps[t]).sum())
    if n_cross:
        raise AssertionError(f"{n_cross} edges cross time steps; full-graph message passing would leak across splits")


def tune_threshold(y_val: np.ndarray, p_val: np.ndarray) -> Tuple[float, float]:
    """Threshold maximising illicit-class F1 on validation (first maximum on a 0.01 grid)."""
    best_t, best_f = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_illicit": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def neighbor_aggregate_features(X: np.ndarray, edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """Non-learned graph features: in-degree, out-degree and the mean of X over undirected neighbours.

    Elliptic edges are intra-time-step, so a node only aggregates same-step transactions.
    Nodes without neighbours get zeros for the mean block.

    Order invariance: the neighbour *sum* of float32 inputs is accumulated in float64 (exact unless
    the summands span ~29+ binary orders of magnitude) and divided by the degree once. The earlier
    version scaled every term by 1/deg before summing, so rounding depended on node order and the
    same snapshot sent in a different order (e.g. interleaved Kafka partitions) produced different
    features and, for borderline nodes, different tree decisions.
    """
    s, t = edge_index
    ones = np.ones(len(s), dtype=np.float64)
    A = sp.coo_matrix((ones, (t, s)), shape=(n_nodes, n_nodes)).tocsr()  # row = receiver
    indeg = np.asarray(A.sum(axis=1)).ravel()
    outdeg = np.asarray(A.sum(axis=0)).ravel()
    U = A + A.T
    U.data[:] = 1.0
    deg = np.asarray(U.sum(axis=1)).ravel()
    sums = U @ np.asarray(X, dtype=np.float64)
    mean = np.divide(sums, deg[:, None], out=np.zeros_like(sums), where=deg[:, None] > 0)
    return np.hstack([indeg[:, None], outdeg[:, None], np.asarray(mean, dtype=np.float32)]).astype(np.float32)


def induced_subgraph(node_mask: np.ndarray, edge_index: np.ndarray):
    """Return (node indices, re-indexed edges) of the subgraph induced by ``node_mask``."""
    idx = np.flatnonzero(node_mask)
    remap = -np.ones(len(node_mask), dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    keep = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    return idx, remap[edge_index[:, keep]]


# --------------------------------------------------------------------------- GAT
def train_gat(X: np.ndarray, mp_edges: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
              val_mask: np.ndarray, seed: int, hidden: int = 32, heads: int = 4, dropout: float = 0.3,
              lr: float = 0.005, weight_decay: float = 5e-4, class_weight: Optional[float] = None,
              max_epochs: int = 200, patience: int = 25):
    """Full-batch GAT training; early stopping and checkpoint selection on validation AP."""
    import torch
    import torch.nn.functional as F
    from src.models.graph_encoder import GraphAttentionEncoder

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GraphAttentionEncoder(X.shape[1], hidden_channels=hidden, heads=heads, dropout=dropout)
    xt = torch.as_tensor(X, dtype=torch.float32)
    et = torch.as_tensor(mp_edges, dtype=torch.long)
    yt = torch.as_tensor(np.where(y < 0, 0, y), dtype=torch.long)
    tr = torch.as_tensor(train_mask)
    weight = torch.tensor([1.0, class_weight], dtype=torch.float32) if class_weight else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(xt, et)[tr], yt[tr], weight=weight)
        loss.backward()
        opt.step()
        p, _ = model.predict(X, mp_edges)
        ap = float(average_precision_score(y[val_mask], p[val_mask]))
        history.append({"epoch": epoch, "loss": round(loss.item(), 5), "val_ap": round(ap, 5)})
        if ap > best_ap:
            best_ap, best_epoch, bad = ap, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_ap": best_ap, "epochs_run": len(history),
                   "train_seconds": time.time() - t0, "history": history}


def cross_fitted_train_embeddings(X, mp_edges, y, time_steps, train_mask, val_mask, seed, gat_kwargs,
                                  n_folds: int = 5):
    """Out-of-fold GAT embeddings for training nodes.

    Training steps 1-29 are cut into ``n_folds`` contiguous step blocks. For each block a
    GAT is trained on the labels of the other blocks and embeds the held-out block, so the
    booster never sees embeddings produced by a network that was fitted on that node's own
    label (classic stacking leakage in the original TGAT-BoostNet). Edges are intra-step,
    so held-out nodes' neighbourhoods contain no training labels from other blocks.
    Only training-set features are affected; validation/test embeddings always come from a
    GAT trained on all of steps 1-29.
    """
    steps = np.arange(TRAIN_STEPS[0], TRAIN_STEPS[1] + 1)
    blocks = np.array_split(steps, n_folds)
    emb_dim = None
    out = None
    for b in blocks:
        hold = np.isin(time_steps, b)
        fold_train = train_mask & ~hold
        model, _ = train_gat(X, mp_edges, y, fold_train, val_mask, seed=seed, **gat_kwargs)
        _, emb = model.predict(X, mp_edges)
        if out is None:
            emb_dim = emb.shape[1]
            out = np.zeros((len(X), emb_dim), dtype=np.float32)
        out[hold] = emb[hold]
    return out
