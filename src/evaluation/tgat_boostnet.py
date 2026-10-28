"""
TGAT-BoostNet as evaluated in notebooks 02/03.

    165 Elliptic features --(4-head GATConv x2, per time-step snapshot)--> 32-d embedding
    [165 features || 32-d embedding] = 197-d --> cost-weighted GBDT (w_illicit = 4.32) --> P(illicit)

Leakage control (see README "Earlier work and corrections"): the GBDT's *training* rows get
cross-fitted embeddings. Steps 1-34 are cut into 5 contiguous step blocks; for each block a GAT is
trained on the other blocks and embeds the held-out block, so the booster never learns from an
embedding produced by a network that saw that row's label. Test embeddings come from a GAT trained
on all of steps 1-34. Edges never cross time steps, so training on the subgraph of the training
steps is exact.

Every setting below is fixed a priori (no validation or test tuning). The GAT uses the class-weighted
cross-entropy convention of Weber et al. (2019) for GNNs on Elliptic (0.3 / 0.7).
"""
import io
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from src.evaluation.data import TEST_STEPS, TRAIN_STEPS, Elliptic, bidirectional, induced

DEFAULT_GBDT = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0)


@dataclass
class TGATBoostNetConfig:
    heads: int = 4
    hidden: int = 32            # channels per head in the first GATConv
    emb_dim: int = 32           # output of the second GATConv = fused embedding size
    dropout: float = 0.3
    lr: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 100
    gnn_class_weight: tuple = (0.3, 0.7)
    n_folds: int = 5
    w_illicit: float = 4.32     # GBDT cost weight of the illicit class (scale_pos_weight)
    use_gat: bool = True        # False -> GBDT on the 165 features only ("no GAT" ablation)
    temporal: bool = False      # True -> add the time step to GAT input (t/49) and GBDT input
    standardise_gat_input: bool = True
    gbdt: dict = field(default_factory=lambda: dict(DEFAULT_GBDT))

    def to_dict(self):
        return asdict(self)


def get_device(prefer_gpu: bool = True) -> str:
    import torch
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"


def make_xgb(params: dict, w_illicit: float, seed: int, device: str = "cpu", n_jobs: int = -1):
    import xgboost as xgb
    return xgb.XGBClassifier(**params, scale_pos_weight=w_illicit, random_state=seed, tree_method="hist",
                             device="cuda" if device == "cuda" else "cpu", n_jobs=n_jobs,
                             eval_metric="logloss")


def xgb_size_and_nodes(model) -> tuple:
    booster = model.get_booster()
    raw = booster.save_raw("ubj")
    n_nodes = sum(len([ln for ln in t.split("\n") if ln.strip()]) for t in booster.get_dump())
    return len(raw), int(n_nodes)


def torch_size_and_params(model) -> tuple:
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell(), int(sum(p.numel() for p in model.parameters()))


def train_gat(X: np.ndarray, edges: np.ndarray, y: np.ndarray, node_mask: np.ndarray, cfg: TGATBoostNetConfig,
              seed: int, device: str = "cpu"):
    """Train a GraphAttentionEncoder on the subgraph induced by ``node_mask`` for cfg.epochs (no early stop).

    Loss uses labelled nodes of that subgraph only. Returns (model, loss history).
    """
    import torch
    import torch.nn.functional as F
    from src.models.graph_encoder import GraphAttentionEncoder

    torch.manual_seed(seed)
    np.random.seed(seed)
    idx, e = induced(node_mask, edges)
    xt = torch.as_tensor(X[idx], dtype=torch.float32, device=device)
    et = torch.as_tensor(e, dtype=torch.long, device=device)
    ys = y[idx]
    lab = torch.as_tensor(ys >= 0, device=device)
    yt = torch.as_tensor(np.where(ys < 0, 0, ys), dtype=torch.long, device=device)
    model = GraphAttentionEncoder(X.shape[1], hidden_channels=cfg.hidden, heads=cfg.heads, dropout=cfg.dropout,
                                  embedding_dim=cfg.emb_dim).to(device)
    w = torch.tensor(cfg.gnn_class_weight, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist = []
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(xt, et)[lab], yt[lab], weight=w)
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
    model.eval()
    return model, hist


def gat_embed(model, X: np.ndarray, edges: np.ndarray, node_mask: np.ndarray, device: str = "cpu"):
    """(node positions, P_gat(illicit), embedding) for the subgraph induced by ``node_mask``."""
    import torch
    import torch.nn.functional as F
    idx, e = induced(node_mask, edges)
    with torch.no_grad():
        xt = torch.as_tensor(X[idx], dtype=torch.float32, device=device)
        et = torch.as_tensor(e, dtype=torch.long, device=device)
        emb = model.embed(xt, et)
        prob = F.softmax(model.out(emb), dim=1)[:, 1]
    return idx, prob.cpu().numpy(), emb.cpu().numpy()


def _standardise_on(X: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Same arithmetic as data.standardise, with statistics from ``fit_rows``."""
    mu = X[fit_rows].astype(np.float64).mean(0)
    sd = X[fit_rows].astype(np.float64).std(0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def fit_tgat_boostnet(data: Elliptic, cfg: Optional[TGATBoostNetConfig] = None, seed: int = 42,
                      device: str = "cpu", log=print, train_steps=None, eval_steps=None) -> dict:
    """Fit on labelled steps 1-34 and score every node of steps 35-49.

    ``train_steps`` / ``eval_steps`` override the split (the tuning stage uses 1-29 / 30-34); the
    standardisation statistics always come from the labelled rows of ``train_steps``.

    Returns a dict with test probabilities of the hybrid (``p_test``) and of the GAT head alone
    (``p_gat_test``), cross-fitted training embeddings, test embeddings, timings, model objects
    and an ``infer(step)`` closure that scores one complete time-step snapshot (for latency).
    """
    cfg = cfg or TGATBoostNetConfig()
    train_steps = tuple(train_steps or TRAIN_STEPS)
    eval_steps = tuple(eval_steps or TEST_STEPS)
    t_start = time.time()
    tr, te = data.mask(train_steps), data.mask(eval_steps)
    Xs = _standardise_on(data.X_raw, tr)
    X_gat = Xs if cfg.standardise_gat_input else data.X_raw.astype(np.float32)
    X_tab = Xs
    if cfg.temporal:
        X_gat = np.hstack([X_gat, (data.time_steps / 49.0).astype(np.float32)[:, None]])
        X_tab = np.hstack([X_tab, data.time_steps.astype(np.float32)[:, None]])
    E = bidirectional(data.edge_index)
    steps = data.time_steps
    in_train_steps = (steps >= train_steps[0]) & (steps <= train_steps[1])
    test_steps = (steps >= eval_steps[0]) & (steps <= eval_steps[1])
    out = {"config": cfg.to_dict(), "seed": seed, "device": device}

    emb_all = None
    if cfg.use_gat:
        t0 = time.time()
        gat_full, hist = train_gat(X_gat, E, data.y, in_train_steps, cfg, seed, device)
        out["gat_full_seconds"] = time.time() - t0
        out["gat_loss_history"] = hist
        emb_all = np.zeros((data.n, cfg.emb_dim), dtype=np.float32)
        p_gat_all = np.zeros(data.n, dtype=np.float32)
        idx, p, emb = gat_embed(gat_full, X_gat, E, test_steps, device)
        emb_all[idx], p_gat_all[idx] = emb, p
        t0 = time.time()
        blocks = np.array_split(np.arange(train_steps[0], train_steps[1] + 1), cfg.n_folds)
        for k, b in enumerate(blocks):
            hold = np.isin(steps, b)
            m_k, _ = train_gat(X_gat, E, data.y, in_train_steps & ~hold, cfg, seed, device)
            idx, p, emb = gat_embed(m_k, X_gat, E, hold, device)
            emb_all[idx], p_gat_all[idx] = emb, p
            log(f"    cross-fit fold {k + 1}/{cfg.n_folds} (held-out steps {b[0]}-{b[-1]}) done")
        out["crossfit_seconds"] = time.time() - t0
        out["p_gat_test"] = p_gat_all[te]
        out["emb_train"], out["emb_test"] = emb_all[tr], emb_all[te]
        out["gat_model"] = gat_full
        F_all = np.hstack([X_tab, emb_all])
    else:
        gat_full = None
        F_all = X_tab
    t0 = time.time()
    booster = make_xgb(cfg.gbdt, cfg.w_illicit, seed, device)
    booster.fit(F_all[tr], data.y[tr])
    out["gbdt_seconds"] = time.time() - t0
    out["p_test"] = booster.predict_proba(F_all[te])[:, 1]
    out["booster"] = booster
    out["train_seconds"] = time.time() - t_start
    out["n_features"] = F_all.shape[1]

    size_b, n_nodes = xgb_size_and_nodes(booster)
    out["size_bytes"], out["n_params"] = size_b, n_nodes
    out["n_params_note"] = "GBDT tree nodes"
    if gat_full is not None:
        gb, gp = torch_size_and_params(gat_full)
        out["size_bytes"] += gb
        out["n_params"] += gp
        out["gat_params"] = gp
        out["n_params_note"] = "GAT weights + GBDT tree nodes"

    def infer(step: int) -> np.ndarray:
        m = steps == step
        if gat_full is None:
            return booster.predict_proba(X_tab[m])[:, 1]
        idx, _, emb = gat_embed(gat_full, X_gat, E, m, device)
        return booster.predict_proba(np.hstack([X_tab[idx], emb]))[:, 1]

    def infer_gat(step: int) -> np.ndarray:
        return gat_embed(gat_full, X_gat, E, steps == step, device)[1]

    out["infer"] = infer
    out["infer_gat"] = infer_gat if gat_full is not None else None
    out["X_tab"], out["X_gat"], out["edges"] = X_tab, X_gat, E
    return out
