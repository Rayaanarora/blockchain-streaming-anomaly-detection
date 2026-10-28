"""
Validation-only fine-tuning of TGAT-BoostNet and of the strongest baselines (notebook 04).

Protocol
  * Inner split inside the training period: fit = labelled steps 1-29, validation = labelled steps 30-34.
    Test steps 35-49 are never touched during the search.
  * Same budget for every family: N_TRIALS random configurations from a fixed space (seeded), ranked by
    validation PR-AUC (threshold-free); the decision threshold is then the validation-F1 maximiser of the
    selected configuration.
  * TGAT-BoostNet has a second, GAT-level stage before its GBDT search: a small grid of GAT settings, each
    scored with cross-fitted embeddings inside steps 1-29 and the default GBDT.
  * Final models: selected configuration refitted on labelled steps 1-34, 5 seeds, test scored once with
    the validation threshold. Saved as results/runs/<family>_tuned/seed_<s> with the threshold in the meta.
"""
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from src.evaluation import baselines as B
from src.evaluation.data import SEEDS, Elliptic
from src.evaluation.tgat_boostnet import (DEFAULT_GBDT, TGATBoostNetConfig, fit_tgat_boostnet, make_xgb,
                                          _standardise_on)

FIT_STEPS, VAL_STEPS = (1, 29), (30, 34)
N_TRIALS = int(os.environ.get("BSAD_TUNING_TRIALS", "40"))
TUNE_DIR = lambda: os.path.join(B.RESULTS_DIR, "tuning")  # noqa: E731
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)

GBDT_SPACE = {"n_estimators": [200, 300, 500, 800], "max_depth": [3, 4, 5, 6, 8], "learning_rate": [0.02, 0.03, 0.05, 0.1],
              "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.4, 0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5],
              "reg_lambda": [0.5, 1.0, 2.0, 5.0], "w_illicit": [1.0, 2.0, 4.32, 7.63]}
LGBM_SPACE = {"n_estimators": [100, 200, 400, 800], "num_leaves": [15, 31, 63, 127], "learning_rate": [0.02, 0.05, 0.1],
              "min_child_samples": [5, 10, 20, 50], "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
              "reg_lambda": [0.0, 1.0, 5.0], "scale_pos_weight": [1.0, 2.0, 4.32, 7.63]}
RF_SPACE = {"n_estimators": [50, 100, 200, 400], "max_features": [0.1, 0.2, 0.3, "sqrt", 50],
            "max_depth": [None, 10, 20, 30], "min_samples_leaf": [1, 2, 5],
            "class_weight": [None, "balanced", "balanced_subsample"]}
SPACES = {"tgat_boostnet": GBDT_SPACE, "xgb": GBDT_SPACE, "lgbm": LGBM_SPACE, "rf": RF_SPACE}
SEARCH_SEEDS = {"tgat_boostnet": 101, "xgb": 101, "lgbm": 202, "rf": 303}  # xgb shares ours: identical GBDT candidates
GAT_GRID = [dict(heads=h, epochs=e) for h in (4, 8) for e in (100, 200)]


def sample_configs(space: dict, n: int, seed: int = 0) -> List[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({k: v[rng.integers(len(v))] for k, v in space.items()})
    return out


def best_threshold(y, p):
    f = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in THRESHOLDS]
    i = int(np.argmax(f))
    return float(THRESHOLDS[i]), float(f[i])


def _clean(d):
    return {k: (None if v is None else (v.item() if hasattr(v, "item") else v)) for k, v in d.items()}


def make_model(family: str, params: dict, seed: int, device: str = "cpu"):
    """Instantiate a candidate; ``params`` comes from SPACES[family]."""
    p = dict(params)
    if family in ("tgat_boostnet", "xgb"):
        w = p.pop("w_illicit")
        return make_xgb(p, w, seed, device, B._n_jobs())
    if family == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**p, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=B._n_jobs())
    if family == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**p, random_state=seed, n_jobs=B._n_jobs())
    raise KeyError(family)


def _cache(name):
    os.makedirs(TUNE_DIR(), exist_ok=True)
    return os.path.join(TUNE_DIR(), name)


# --------------------------------------------------------------------------- inner-split features
def inner_tabular(data: Elliptic):
    fit, val = data.mask(FIT_STEPS), data.mask(VAL_STEPS)
    X = _standardise_on(data.X_raw, fit)
    return X[fit], data.y[fit], X[val], data.y[val]


def inner_tgat_features(data: Elliptic, gat_kw: dict, seed: int = 42, device: str = "cpu", log=print):
    """[165 || cross-fitted GAT embedding] for fit rows (1-29) and [165 || embedding] for val rows (30-34)."""
    cfg = TGATBoostNetConfig(**gat_kw)
    r = fit_tgat_boostnet(data, cfg, seed, device, log=log, train_steps=FIT_STEPS, eval_steps=VAL_STEPS)
    fit, val = data.mask(FIT_STEPS), data.mask(VAL_STEPS)
    Xs = r["X_tab"]
    return (np.hstack([Xs[fit], r["emb_train"]]), data.y[fit], np.hstack([Xs[val], r["emb_test"]]), data.y[val],
            float(average_precision_score(data.y[val], r["p_test"])))


# --------------------------------------------------------------------------- search stages
def gat_stage(data: Elliptic, device: str = "cpu", log=print, seed: int = 42) -> pd.DataFrame:
    path = _cache("gat_stage.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    rows = []
    for g in GAT_GRID:
        t0 = time.time()
        _, _, _, _, ap = inner_tgat_features(data, g, seed, device, log=lambda *a: None)
        rows.append(dict(g, val_pr_auc=ap, seconds=time.time() - t0))
        log(f"  GAT {g}: validation PR-AUC (default GBDT) = {ap:.4f}  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def random_search(family: str, Xf, yf, Xv, yv, device: str = "cpu", log=print, seed: int = 42) -> pd.DataFrame:
    path = _cache(f"search_{family}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    rows = []
    for i, params in enumerate(sample_configs(SPACES[family], N_TRIALS, seed=SEARCH_SEEDS[family])):
        t0 = time.time()
        m = make_model(family, params, seed, device)
        m.fit(Xf, yf)
        p = m.predict_proba(Xv)[:, 1]
        thr, f1v = best_threshold(yv, p)
        rows.append(dict(trial=i, **{k: (str(v) if v is None or isinstance(v, str) else v) for k, v in params.items()},
                         val_pr_auc=float(average_precision_score(yv, p)), val_f1_at_best_t=f1v, best_threshold=thr,
                         seconds=time.time() - t0))
        if (i + 1) % 10 == 0:
            log(f"  {family}: {i + 1}/{N_TRIALS} trials, best val PR-AUC so far {max(r['val_pr_auc'] for r in rows):.4f}")
    df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    df.to_csv(path, index=False)
    return df


def _parse(v):
    if hasattr(v, "item"):  # numpy scalar -> Python scalar
        v = v.item()
    if isinstance(v, str):
        if v == "None":
            return None
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def selected(family: str, search: pd.DataFrame) -> dict:
    """Best row -> (params, threshold). Ties in val PR-AUC broken by val F1."""
    best = search.sort_values(["val_pr_auc", "val_f1_at_best_t"], ascending=False).iloc[0]
    params = {k: _parse(best[k]) for k in SPACES[family]}
    if family in ("tgat_boostnet", "xgb"):
        params["w_illicit"] = float(best["w_illicit"])
        params["learning_rate"] = float(best["learning_rate"])
        for k in ("subsample", "colsample_bytree", "reg_lambda"):
            params[k] = float(best[k])
    if family == "lgbm":
        for k in ("learning_rate", "subsample", "colsample_bytree", "reg_lambda", "scale_pos_weight"):
            params[k] = float(best[k])
    return {"params": _clean(params), "threshold": float(best["best_threshold"]), "val_pr_auc": float(best["val_pr_auc"]),
            "val_f1": float(best["val_f1_at_best_t"])}


def tune_all(data: Elliptic, device: str = "cpu", log=print) -> Dict[str, dict]:
    """Run (or load) every search stage and return the selected configuration per family."""
    sel_path = _cache("selected.json")
    if os.path.exists(sel_path):
        return json.load(open(sel_path))
    out = {}
    log("TGAT-BoostNet stage 1: GAT grid (validation PR-AUC of the full hybrid with the default GBDT)")
    g = gat_stage(data, device, log)
    gbest = g.sort_values("val_pr_auc", ascending=False).iloc[0]
    gat_kw = {"heads": int(gbest.heads), "epochs": int(gbest.epochs)}
    log(f"  selected GAT {gat_kw}")
    log("TGAT-BoostNet stage 2: GBDT random search on the selected GAT's embeddings")
    Xf, yf, Xv, yv, _ = inner_tgat_features(data, gat_kw, 42, device, log=lambda *a: None)
    s = selected("tgat_boostnet", random_search("tgat_boostnet", Xf, yf, Xv, yv, device, log))
    out["tgat_boostnet"] = dict(s, gat=gat_kw)
    Xf, yf, Xv, yv = inner_tabular(data)
    for fam in ("lgbm", "rf", "xgb"):
        log(f"{fam}: random search")
        out[fam] = selected(fam, random_search(fam, Xf, yf, Xv, yv, device, log))
    json.dump(out, open(sel_path, "w"), indent=1)
    return out


# --------------------------------------------------------------------------- final 5-seed refits
def final_run(family: str, sel: dict, data: Elliptic, seed: int, device: str = "cpu", force: bool = False, log=print):
    """Refit the selected configuration on labelled steps 1-34 and score test 35-49 once."""
    key = f"{family}_tuned"
    if not force and B.has_run(key, seed):
        return B.load_run(key, seed, data)
    meta = dict(device=device, threshold=sel["threshold"], selected_params=_clean(sel["params"]),
                protocol="validation-tuned on 1-29/30-34; refit 1-34; test 35-49 scored once at the validation threshold")
    if family == "tgat_boostnet":
        params = dict(sel["params"])
        w = params.pop("w_illicit")
        cfg = TGATBoostNetConfig(**sel["gat"], w_illicit=w, gbdt=params)
        outs, _ = B.run_tgat_family(data, seed, device, cfg, key=key, log=log)
        v = outs[key]
        p = v.pop("p_test")
        B.save_run(key, seed, data, p, dict(meta, **{k: x for k, x in v.items() if k != "config"},
                                            config=asdict(cfg)))
    else:
        X = _standardise_on(data.X_raw, data.train_mask)
        m = make_model(family, sel["params"], seed, device)
        t0 = time.time()
        m.fit(X[data.train_mask], data.y[data.train_mask])
        tr_s = time.time() - t0
        size, n_params, note = B.model_size_and_params(m)
        lat = B.measure_latency(lambda s: m.predict_proba(X[data.time_steps == s])[:, 1], data)
        B.save_run(key, seed, data, m.predict_proba(X[data.test_mask])[:, 1],
                   dict(meta, train_seconds=tr_s, size_bytes=size, n_params=n_params, n_params_note=note, **lat))
    return B.load_run(key, seed, data)


def final_runs(sel: Dict[str, dict], data: Elliptic, device: str = "cpu", log=print) -> Dict[str, Dict[int, dict]]:
    return {f"{fam}_tuned": {s: final_run(fam, sel[fam], data, s, device, log=log) for s in SEEDS} for fam in sel}
