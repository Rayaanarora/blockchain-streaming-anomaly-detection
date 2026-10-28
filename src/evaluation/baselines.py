"""
Reproduced comparison models and the cached experiment runner used by notebooks 02 and 03.

Every model is trained on labelled steps 1-34 and scored on labelled steps 35-49 with threshold
0.5 (see src/evaluation/data.py). Hyper-parameters are the published configuration where one
exists (Weber et al. 2019 for LR / RF / MLP / GCN / Skip-GCN) and library defaults otherwise.
Nothing is tuned on test data.

Results are cached under results/runs/<model>/seed_<seed>.{npz,json} so the notebooks can be
re-run quickly; delete a folder (or pass force=True) to recompute it. The cache is keyed on the
test transaction ids, so a cache built from a different node order is rejected.
"""
import json
import os
import platform
import time
from typing import Callable, Dict, Optional

import numpy as np

from src.evaluation.data import (REPO_ROOT, SEEDS, TRAIN_STEPS, Elliptic, bidirectional, induced,
                                 standardise)
from src.evaluation.tgat_boostnet import (DEFAULT_GBDT, TGATBoostNetConfig, fit_tgat_boostnet, gat_embed,
                                          make_xgb, torch_size_and_params, xgb_size_and_nodes)

RESULTS_DIR = os.environ.get("BSAD_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
OURS = "tgat_boostnet"

# key -> (display name, family, stochastic?)
MODELS: Dict[str, tuple] = {
    "tgat_boostnet": ("TGAT-BoostNet (ours)", "Hybrid GNN+GBDT", True),
    "lr": ("Logistic Regression", "Linear", False),
    "svm": ("SVM (RBF)", "Kernel", True),
    "rf": ("Random Forest", "Tree ensemble", True),
    "xgb": ("XGBoost", "Tree ensemble", True),
    "lgbm": ("LightGBM", "Tree ensemble", True),
    "catboost": ("CatBoost", "Tree ensemble", True),
    "mlp": ("MLP", "Neural net", True),
    "gcn": ("GCN", "GNN", True),
    "skip_gcn": ("Skip-GCN", "GNN", True),
    "graphsage": ("GraphSAGE", "GNN", True),
    "gat": ("GAT (alone)", "GNN", True),
    "evolvegcn": ("EvolveGCN-O", "Temporal GNN", True),
    "gat_rf": ("GAT + RF", "Hybrid GNN+tree", True),
    "gcn_xgb": ("GCN + XGBoost", "Hybrid GNN+tree", True),
    "gbdt165_w": ("GBDT-165 (w=4.32)", "Tree ensemble", True),
    "gbdt197_unw": ("GBDT-197 (no cost weight)", "Hybrid GNN+GBDT", True),
}
COMPARISON = list(MODELS)

# Ablation variants of TGAT-BoostNet (each changes exactly one thing).
ABLATIONS: Dict[str, tuple] = {
    "abl_heads1": ("1 head", dict(heads=1)),
    "abl_heads2": ("2 heads", dict(heads=2)),
    "abl_heads8": ("8 heads", dict(heads=8)),
    "abl_dim16": ("emb dim 16", dict(emb_dim=16)),
    "abl_dim64": ("emb dim 64", dict(emb_dim=64)),
    "abl_temporal": ("+ temporal feature", dict(temporal=True)),
    "abl_rawfeat": ("raw (unscaled) GAT input", dict(standardise_gat_input=False)),
}
for _k, (_lbl, _) in ABLATIONS.items():
    MODELS[_k] = (f"TGAT-BoostNet, {_lbl}", "Ablation", True)

# Which runner produces which keys (a TGAT run also yields the GAT head and two GBDT variants).
TGAT_FAMILY = ("tgat_boostnet", "gat", "gat_rf", "gbdt197_unw")


def display(key: str) -> str:
    return MODELS[key][0]


# --------------------------------------------------------------------------- cache
def _paths(key, seed):
    d = os.path.join(RESULTS_DIR, "runs", key)
    return d, os.path.join(d, f"seed_{seed}.npz"), os.path.join(d, f"seed_{seed}.json")


def _test_signature(data: Elliptic) -> str:
    import hashlib
    return hashlib.sha1(data.tx_ids[data.test_mask].tobytes()).hexdigest()[:16]


def has_run(key: str, seed: int) -> bool:
    return os.path.exists(_paths(key, seed)[1]) and os.path.exists(_paths(key, seed)[2])


def save_run(key: str, seed: int, data: Elliptic, p_test: np.ndarray, meta: dict):
    d, npz, js = _paths(key, seed)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(npz, p_test=np.asarray(p_test, dtype=np.float32))
    meta = dict(meta, model=key, seed=seed, test_signature=_test_signature(data), n_test=int(len(p_test)),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), host=platform.node(), platform=platform.platform())
    with open(js, "w") as f:
        json.dump(meta, f, indent=1, default=float)


def load_run(key: str, seed: int, data: Optional[Elliptic] = None) -> dict:
    _, npz, js = _paths(key, seed)
    with open(js) as f:
        meta = json.load(f)
    if data is not None and meta["test_signature"] != _test_signature(data):
        raise ValueError(f"cached run {key}/seed_{seed} was built on a different test set")
    meta["p_test"] = np.load(npz)["p_test"]
    return meta


def save_embeddings(key, seed, emb_train, emb_test):
    d = os.path.join(RESULTS_DIR, "embeddings", key)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"seed_{seed}.npz"), emb_train=emb_train.astype(np.float16),
                        emb_test=emb_test.astype(np.float16))


def load_embeddings(key, seed):
    z = np.load(os.path.join(RESULTS_DIR, "embeddings", key, f"seed_{seed}.npz"))
    return z["emb_train"].astype(np.float32), z["emb_test"].astype(np.float32)


def has_embeddings(key, seed):
    return os.path.exists(os.path.join(RESULTS_DIR, "embeddings", key, f"seed_{seed}.npz"))


# --------------------------------------------------------------------------- latency
def measure_latency(infer: Callable[[int], np.ndarray], data: Elliptic) -> dict:
    """Score each complete test time-step snapshot (labelled + unlabelled nodes) once after a warm-up.

    A "micro-batch" is one Elliptic time step (~1-6k transactions), the unit the streaming
    scorer receives. Returns median / p95 latency per snapshot and transactions per second.
    """
    steps = list(range(TRAIN_STEPS[1] + 1, 50))
    infer(steps[0])  # warm-up (lazy init, caches)
    times, n = [], 0
    for s in steps:
        t0 = time.perf_counter()
        infer(s)
        times.append(time.perf_counter() - t0)
        n += int((data.time_steps == s).sum())
    times = np.array(times)
    return {"latency_ms_median": float(np.median(times) * 1e3), "latency_ms_p95": float(np.quantile(times, .95) * 1e3),
            "throughput_tx_per_s": float(n / times.sum()), "n_snapshots": len(steps),
            "mean_snapshot_size": float(n / len(steps))}


# --------------------------------------------------------------------------- tabular baselines
def _n_jobs():
    return int(os.environ.get("BSAD_N_JOBS", "-1"))


def _tabular(key: str, seed: int, device: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    import xgboost as xgb
    if key == "lr":  # Weber et al.: scikit-learn defaults (max_iter raised so lbfgs converges)
        return LogisticRegression(max_iter=2000)
    if key == "svm":
        return SVC(kernel="rbf", probability=True, random_state=seed)
    if key == "rf":  # Weber et al.: 50 estimators, 50 max features
        return RandomForestClassifier(n_estimators=50, max_features=50, random_state=seed, n_jobs=_n_jobs())
    if key == "xgb":
        return xgb.XGBClassifier(random_state=seed, tree_method="hist", n_jobs=_n_jobs(),
                                 device="cuda" if device == "cuda" else "cpu")
    if key == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=_n_jobs())
    if key == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, verbose=0, thread_count=_n_jobs(), allow_writing_files=False)
    if key == "mlp":  # Weber et al.: one hidden layer of 50, Adam, lr 0.001, 200 epochs
        return MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.001, max_iter=200, random_state=seed)
    if key == "gbdt165_w":  # TGAT-BoostNet's booster, same settings, without the GAT embedding
        return make_xgb(DEFAULT_GBDT, 4.32, seed, device, _n_jobs())
    raise KeyError(key)


def model_size_and_params(model) -> tuple:
    import pickle
    size = len(pickle.dumps(model))
    name = type(model).__name__
    try:
        if name == "XGBClassifier":
            return xgb_size_and_nodes(model)[0], xgb_size_and_nodes(model)[1], "tree nodes"
        if name == "RandomForestClassifier":
            return size, int(sum(e.tree_.node_count for e in model.estimators_)), "tree nodes"
        if name == "LGBMClassifier":
            df = model.booster_.trees_to_dataframe()
            return size, int(len(df)), "tree nodes"
        if name == "CatBoostClassifier":
            # oblivious trees: 2^depth leaves (+ depth split conditions) per tree
            d = model.get_all_params().get("depth", 6)
            return size, int(model.tree_count_ * (2 ** d + d)), "tree leaves + splits"
        if name == "LogisticRegression":
            return size, int(model.coef_.size + model.intercept_.size), "weights"
        if name == "MLPClassifier":
            return size, int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)), "weights"
        if name == "SVC":
            return size, int(model.support_vectors_.size + model.dual_coef_.size), "support-vector values"
    except Exception:  # noqa: BLE001
        pass
    return size, -1, "n/a"


def run_tabular(key, data: Elliptic, seed: int, device: str = "cpu", X: Optional[np.ndarray] = None,
                quick: bool = False) -> dict:
    Xs = standardise(data)[0] if X is None else X
    tr, te = data.train_mask.copy(), data.test_mask
    if quick and key == "svm":  # smoke tests only: RBF-SVC cost is quadratic in the training size
        keep = np.flatnonzero(tr)[np.random.default_rng(seed).permutation(tr.sum())[3000:]]
        tr[keep] = False
    model = _tabular(key, seed, device)
    t0 = time.time()
    model.fit(Xs[tr], data.y[tr])
    train_s = time.time() - t0
    p = model.predict_proba(Xs[te])[:, 1]
    size, n_params, note = model_size_and_params(model)
    lat = measure_latency(lambda s: model.predict_proba(Xs[data.time_steps == s])[:, 1], data)
    return dict(p_test=p, train_seconds=train_s, size_bytes=size, n_params=n_params, n_params_note=note, **lat)


# --------------------------------------------------------------------------- GNN baselines
def _gnn_module(kind: str, in_dim: int, hidden: int = 100):
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            conv = SAGEConv if kind == "graphsage" else GCNConv
            self.conv1 = conv(in_dim, hidden)
            self.conv2 = conv(hidden, 2)
            self.skip = nn.Linear(in_dim, 2) if kind == "skip_gcn" else None

        def embed(self, x, e):
            return F.relu(self.conv1(x, e))

        def forward(self, x, e):
            out = self.conv2(self.embed(x, e), e)
            return out + self.skip(x) if self.skip is not None else out
    return Net()


def train_gnn(kind, X, E, y, node_mask, seed, device, epochs=1000, lr=0.001, weight=(0.3, 0.7)):
    """Weber et al. (2019) GCN recipe: 2 layers, 100-d embedding, Adam lr 0.001, 1000 epochs, CE 0.3/0.7."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    idx, e = induced(node_mask, E)
    xt = torch.as_tensor(X[idx], dtype=torch.float32, device=device)
    et = torch.as_tensor(e, dtype=torch.long, device=device)
    ys = y[idx]
    lab = torch.as_tensor(ys >= 0, device=device)
    yt = torch.as_tensor(np.where(ys < 0, 0, ys), dtype=torch.long, device=device)
    model = _gnn_module(kind, X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = torch.tensor(weight, dtype=torch.float32, device=device)
    hist = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(xt, et)[lab], yt[lab], weight=w)
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
    model.eval()
    return model, hist


def gnn_predict(model, X, E, node_mask, device):
    import torch
    import torch.nn.functional as F
    idx, e = induced(node_mask, E)
    with torch.no_grad():
        xt = torch.as_tensor(X[idx], dtype=torch.float32, device=device)
        et = torch.as_tensor(e, dtype=torch.long, device=device)
        emb = model.embed(xt, et)
        p = F.softmax(model(xt, et), dim=1)[:, 1]
    return idx, p.cpu().numpy(), emb.cpu().numpy()


def run_gnn(kind, data: Elliptic, seed: int, device: str = "cpu", epochs: int = 1000) -> dict:
    Xs = standardise(data)[0]
    E = bidirectional(data.edge_index)
    steps = data.time_steps
    t0 = time.time()
    model, hist = train_gnn(kind, Xs, E, data.y, steps <= TRAIN_STEPS[1], seed, device, epochs=epochs)
    train_s = time.time() - t0
    idx, p, _ = gnn_predict(model, Xs, E, steps > TRAIN_STEPS[1], device)
    pos = np.full(data.n, np.nan, dtype=np.float32)
    pos[idx] = p
    size, n_params = torch_size_and_params(model)
    lat = measure_latency(lambda s: gnn_predict(model, Xs, E, steps == s, device)[1], data)
    return dict(p_test=pos[data.test_mask], train_seconds=train_s, size_bytes=size, n_params=n_params,
                n_params_note="weights", loss_history=hist, **lat)


def run_gcn_xgb(data: Elliptic, seed: int, device: str = "cpu", epochs: int = 1000, n_folds: int = 5, log=print):
    """XGBoost (defaults) on [165 features || 100-d GCN embedding]; training rows get cross-fitted embeddings."""
    import xgboost as xgb
    Xs = standardise(data)[0]
    E = bidirectional(data.edge_index)
    steps = data.time_steps
    in_tr = steps <= TRAIN_STEPS[1]
    t0 = time.time()
    full, _ = train_gnn("gcn", Xs, E, data.y, in_tr, seed, device, epochs=epochs)
    emb = np.zeros((data.n, 100), dtype=np.float32)
    idx, _, em = gnn_predict(full, Xs, E, ~in_tr, device)
    emb[idx] = em
    for k, b in enumerate(np.array_split(np.arange(TRAIN_STEPS[0], TRAIN_STEPS[1] + 1), n_folds)):
        hold = np.isin(steps, b)
        m_k, _ = train_gnn("gcn", Xs, E, data.y, in_tr & ~hold, seed, device, epochs=epochs)
        idx, _, em = gnn_predict(m_k, Xs, E, hold, device)
        emb[idx] = em
        log(f"    GCN cross-fit fold {k + 1}/{n_folds} done")
    F_all = np.hstack([Xs, emb])
    tr, te = data.train_mask, data.test_mask
    clf = xgb.XGBClassifier(random_state=seed, tree_method="hist", n_jobs=_n_jobs(),
                            device="cuda" if device == "cuda" else "cpu")
    clf.fit(F_all[tr], data.y[tr])
    train_s = time.time() - t0
    p = clf.predict_proba(F_all[te])[:, 1]
    gs, gp = torch_size_and_params(full)
    xs, xn = xgb_size_and_nodes(clf)

    def infer(s):
        idx, _, em = gnn_predict(full, Xs, E, steps == s, device)
        return clf.predict_proba(np.hstack([Xs[idx], em]))[:, 1]
    return dict(p_test=p, train_seconds=train_s, size_bytes=gs + xs, n_params=gp + xn,
                n_params_note="GCN weights + tree nodes", **measure_latency(infer, data))


# --------------------------------------------------------------------------- EvolveGCN-O
def run_evolvegcn(data: Elliptic, seed: int, device: str = "cpu", epochs: int = 300, hidden: int = 100,
                  lr: float = 0.005, weight=(0.3, 0.7)) -> dict:
    """EvolveGCN-O (Pareja et al., AAAI 2020): per-step 2-layer GCN whose weight matrices are evolved
    by a GRU from one time step to the next. Trained on the sequence of steps 1-34 (loss on labelled
    nodes); test steps 35-49 are scored by continuing the weight evolution (label-free).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    Xs = standardise(data)[0]
    E = bidirectional(data.edge_index)
    steps = data.time_steps

    snaps = []
    for s in range(1, 50):
        m = steps == s
        idx, e = induced(m, E)
        n = len(idx)
        loops = np.arange(n)
        r = np.concatenate([e[1], loops])
        c = np.concatenate([e[0], loops])
        deg = np.bincount(r, minlength=n).astype(np.float32)
        val = 1.0 / np.sqrt(deg[r] * deg[c])
        A = torch.sparse_coo_tensor(np.vstack([r, c]), val.astype(np.float32), (n, n)).coalesce().to(device)
        ys = data.y[idx]
        snaps.append((idx, torch.as_tensor(Xs[idx], device=device), A,
                      torch.as_tensor(np.where(ys < 0, 0, ys), device=device), torch.as_tensor(ys >= 0, device=device)))

    in_dim = Xs.shape[1]

    class EvolveGCNO(nn.Module):
        def __init__(self):
            super().__init__()
            self.W1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(in_dim, hidden)))
            self.W2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden, hidden)))
            self.gru1 = nn.GRUCell(in_dim, in_dim)   # evolves each column of W1
            self.gru2 = nn.GRUCell(hidden, hidden)
            self.out = nn.Linear(hidden, 2)

        def forward(self, seq):
            W1, W2, outs = self.W1, self.W2, []
            for (_, X, A, _, _) in seq:
                W1 = self.gru1(W1.t(), W1.t()).t()
                W2 = self.gru2(W2.t(), W2.t()).t()
                h = F.relu(torch.sparse.mm(A, X @ W1))
                h = F.relu(torch.sparse.mm(A, h @ W2))
                outs.append(self.out(h))
            return outs

    model = EvolveGCNO().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = torch.tensor(weight, dtype=torch.float32, device=device)
    train_seq = snaps[:TRAIN_STEPS[1]]
    hist = []
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        outs = model(train_seq)
        logits = torch.cat([o[s[4]] for o, s in zip(outs, train_seq)])
        target = torch.cat([s[3][s[4]] for s in train_seq])
        loss = F.cross_entropy(logits, target, weight=w)
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
    train_s = time.time() - t0
    model.eval()
    with torch.no_grad():
        outs = model(snaps)
    pos = np.full(data.n, np.nan, dtype=np.float32)
    for (idx, *_), o in zip(snaps, outs):
        pos[idx] = F.softmax(o, dim=1)[:, 1].cpu().numpy()
    size, n_params = torch_size_and_params(model)

    def infer(s):  # streaming: evolve weights through all steps up to s, then score step s
        with torch.no_grad():
            return F.softmax(model(snaps[:s])[-1], dim=1)[:, 1].cpu().numpy()
    return dict(p_test=pos[data.test_mask], train_seconds=train_s, size_bytes=size, n_params=n_params,
                n_params_note="weights", loss_history=hist, **measure_latency(infer, data))


# --------------------------------------------------------------------------- TGAT family
def run_tgat_family(data: Elliptic, seed: int, device: str = "cpu", cfg: Optional[TGATBoostNetConfig] = None,
                    key: str = OURS, log=print, save_artifacts: bool = False) -> Dict[str, dict]:
    """Fit TGAT-BoostNet once; also derive GAT-alone, GAT+RF and GBDT-197-unweighted from the same
    GAT embeddings (identical seed and embeddings, so these comparisons isolate one component)."""
    from sklearn.ensemble import RandomForestClassifier
    r = fit_tgat_boostnet(data, cfg, seed, device, log=log)
    tr, te = data.train_mask, data.test_mask
    steps = data.time_steps
    common = {k: r[k] for k in ("gat_full_seconds", "crossfit_seconds", "gbdt_seconds", "n_features", "gat_params")
              if k in r}
    res = {key: dict(p_test=r["p_test"], train_seconds=r["train_seconds"], size_bytes=r["size_bytes"],
                     n_params=r["n_params"], n_params_note=r["n_params_note"], config=r["config"],
                     gat_loss_history=r.get("gat_loss_history"), **common, **measure_latency(r["infer"], data))}
    if key != OURS or not r["config"]["use_gat"]:
        return res, r
    save_embeddings(OURS, seed, r["emb_train"], r["emb_test"])
    gat_model, X_gat, X_tab, E = r["gat_model"], r["X_gat"], r["X_tab"], r["edges"]
    gs, gp = torch_size_and_params(gat_model)
    res["gat"] = dict(p_test=r["p_gat_test"], train_seconds=r["gat_full_seconds"], size_bytes=gs, n_params=gp,
                      n_params_note="weights", **measure_latency(r["infer_gat"], data))
    F_tr = np.hstack([X_tab[tr], r["emb_train"]])
    F_te = np.hstack([X_tab[te], r["emb_test"]])

    def with_emb(clf):
        def infer(s):
            idx, _, emb = gat_embed(gat_model, X_gat, E, steps == s, device)
            return clf.predict_proba(np.hstack([X_tab[idx], emb]))[:, 1]
        return infer
    emb_seconds = r["gat_full_seconds"] + r["crossfit_seconds"]
    for k, clf in (("gat_rf", RandomForestClassifier(n_estimators=50, max_features=50, random_state=seed,
                                                     n_jobs=_n_jobs())),
                   ("gbdt197_unw", make_xgb(DEFAULT_GBDT, 1.0, seed, device, _n_jobs()))):
        t0 = time.time()
        clf.fit(F_tr, data.y[tr])
        fit_s = time.time() - t0
        size, n_params, note = model_size_and_params(clf)
        res[k] = dict(p_test=clf.predict_proba(F_te)[:, 1], train_seconds=emb_seconds + fit_s,
                      size_bytes=size + gs, n_params=n_params + gp, n_params_note=f"GAT weights + {note}",
                      **measure_latency(with_emb(clf), data))
    if save_artifacts:
        import torch
        d = os.path.join(RESULTS_DIR, "artifacts", f"{OURS}_seed{seed}")
        os.makedirs(d, exist_ok=True)
        torch.save({"config": gat_model.config, "state_dict": gat_model.state_dict()}, os.path.join(d, "gat.pt"))
        r["booster"].get_booster().save_model(os.path.join(d, "booster.ubj"))
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(r["config"], f, indent=1)
    return res, r


# --------------------------------------------------------------------------- runner
def run(key: str, data: Elliptic, seed: int, device: str = "cpu", force: bool = False, log=print,
        quick: Optional[bool] = None) -> dict:
    """Return the cached run for (key, seed), computing (and caching) it first if needed.

    ``quick`` shrinks epochs for smoke tests (default: $BSAD_QUICK == "1"); smoke runs must use a
    separate cache (the notebooks and the driver point BSAD_RESULTS_DIR at results_quick/).
    """
    if quick is None:
        quick = os.environ.get("BSAD_QUICK") == "1"
    if not force and has_run(key, seed):
        return load_run(key, seed, data)
    log(f"[{time.strftime('%H:%M:%S')}] training {key} seed={seed} on {device}")
    meta_common = dict(device=device, quick=quick, torch_threads=_torch_threads(),
                       protocol="train labelled steps 1-34, test labelled steps 35-49, threshold 0.5")
    ep = 3 if quick else None
    if key in TGAT_FAMILY or key in ABLATIONS:
        if key in ABLATIONS:
            cfg = TGATBoostNetConfig(**ABLATIONS[key][1])
            run_key = key
        else:
            cfg, run_key = TGATBoostNetConfig(), OURS
        if quick:
            cfg.epochs = ep
        outs, _ = run_tgat_family(data, seed, device, cfg, key=run_key, log=log,
                                  save_artifacts=(seed == SEEDS[0] and run_key == OURS))
        for k, v in outs.items():
            p = v.pop("p_test")
            save_run(k, seed, data, p, dict(meta_common, **v))
        return load_run(key, seed, data)
    if key in ("gcn", "skip_gcn", "graphsage"):
        out = run_gnn(key, data, seed, device, epochs=ep or 1000)
    elif key == "evolvegcn":
        out = run_evolvegcn(data, seed, device, epochs=ep or 300)
    elif key == "gcn_xgb":
        out = run_gcn_xgb(data, seed, device, epochs=ep or 1000, log=log)
    else:
        out = run_tabular(key, data, seed, device, quick=quick)
    p = out.pop("p_test")
    save_run(key, seed, data, p, dict(meta_common, **out))
    return load_run(key, seed, data)


def _torch_threads():
    try:
        import torch
        return torch.get_num_threads()
    except Exception:  # noqa: BLE001
        return None


def seeds_for(key: str):
    """Seeds to run: 5 for stochastic models, 1 for deterministic ones (identical across seeds)."""
    return list(SEEDS) if MODELS[key][2] else [SEEDS[0]]


def collect(keys, data: Elliptic, device="cpu", force=False, log=print) -> Dict[str, Dict[int, dict]]:
    """{model: {seed: run}} for all requested models, computing missing runs."""
    return {k: {s: run(k, data, s, device, force, log) for s in seeds_for(k)} for k in keys}
