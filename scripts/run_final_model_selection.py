"""
Final Elliptic model selection (Phase 2).

Protocol: train = labelled steps 1-29, validation = labelled steps 30-34,
test = labelled steps 35-49. Hyper-parameters, early stopping, thresholds and the
final model choice use validation data only. Test labels are read exactly once per
fitted model, after its threshold has been fixed.

Stage A (seed 42): small validation-AP grid searches for HistGBDT, XGBoost and GAT.
Stage B (seeds 42, 123, 456, 789, 2024): every configuration below, with validation
thresholds, test metrics, confusion matrices and per-step metrics.

  hgb_165_legacy      HistGBDT exactly as in the earlier ablation (internal random early stopping)
  hgb_165_tuned       HistGBDT, grid-selected, early stopping off (deterministic)
  xgb_165_legacy      XGBoost exactly as in the earlier ablation
  xgb_165_tuned       XGBoost, grid-selected
  rf_165              Random forest reference
  hgb_lf93_tuned      HistGBDT on local features f1-f93 only
  hgb_165_nbr         HistGBDT on 165 features + non-learned neighbour aggregates (95)
  gat_alone           GraphAttentionEncoder classifier (grid-selected)
  xgb_165_gat_cf      XGBoost on 165 + 32 cross-fitted GAT embeddings
  hgb_165_gat_cf      HistGBDT on 165 + 32 cross-fitted GAT embeddings
  hgb_165_gat_insample HistGBDT on 165 + 32 in-sample GAT embeddings (stacking-leakage control)
  ens_hgb_hgbgat      mean probability of hgb_165_tuned and hgb_165_gat_cf

Selection rule (decided before looking at test): highest mean validation F1 over the
five seeds; ties (|diff| < 0.005) broken by mean validation AP, then by simplicity
(fewer components).

Usage:
    .venv\\Scripts\\python.exe scripts\\run_final_model_selection.py
Outputs: data/validation/final_model_selection/, artifacts/models/elliptic/<version>/
"""
import argparse
import json
import os
import sys
import time

# Cap native thread pools BEFORE numpy/sklearn/torch load (oversubscription made fits 10x slower).
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.elliptic_training import (assert_no_cross_step_edges, cross_fitted_train_embeddings,  # noqa: E402
                                          evaluate, induced_subgraph, load_npz, load_spark_output,
                                          neighbor_aggregate_features, train_gat, tune_threshold)
from src.models.graph_encoder import make_edge_index  # noqa: E402

SEEDS = [42, 123, 456, 789, 2024]
COMPONENTS = {  # used for the simplicity tie-break
    "hgb_165_legacy": 1, "hgb_165_tuned": 1, "xgb_165_legacy": 1, "xgb_165_tuned": 1, "rf_165": 1,
    "hgb_lf93_tuned": 1, "hgb_165_nbr": 2, "gat_alone": 1, "xgb_165_gat_cf": 2, "hgb_165_gat_cf": 2,
    "hgb_165_gat_insample": 2, "ens_hgb_hgbgat": 3,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["npz", "spark"], default="spark",
                    help="npz = legacy pandas preprocessing, spark = data/processed/elliptic_spark")
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/processed/elliptic/elliptic_preprocessed.npz"))
    ap.add_argument("--spark-dir", default=os.path.join(ROOT, "data/processed/elliptic_spark"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/validation/final_model_selection"))
    ap.add_argument("--artifact-dir", default=os.path.join(ROOT, "artifacts/models/elliptic"))
    ap.add_argument("--model-version", default="1.0.0")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="tiny grids/epochs for smoke testing")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; aggregate seed_<s>.json / seed_<s>_probs.parquet written by earlier "
                         "(possibly parallel) runs for --seeds, select on validation and export")
    ap.add_argument("--train-only", action="store_true",
                    help="write seed_<s>.json / seed_<s>_probs.parquet for --seeds and stop (no aggregation or "
                         "export); use for parallel seed workers, then run --aggregate-only with all seeds")
    return ap.parse_args()


def hgb(params, seed, ratio):
    return HistGradientBoostingClassifier(class_weight={0: 1.0, 1: ratio}, random_state=seed, **params)


def xgbc(params, seed, ratio):
    return xgb.XGBClassifier(scale_pos_weight=ratio, random_state=seed, eval_metric="logloss",
                             n_jobs=8, tree_method="hist", **params)


def fit_predict(model, Xtr, ytr, Xs):
    t0 = time.time()
    model.fit(Xtr, ytr)
    fit_s = time.time() - t0
    return model, [model.predict_proba(X)[:, 1] for X in Xs], fit_s


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    os.makedirs(args.out, exist_ok=True)
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731

    data = load_npz(args.npz) if args.source == "npz" else load_spark_output(args.spark_dir)
    assert_no_cross_step_edges(data)
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    y = data.y
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
    assert (y[tr | va | te] >= 0).all(), "unknown labels inside a supervised split"
    ratio = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    log(f"source={args.source} nodes={len(y)} edges={data.edge_index.shape[1]} train={tr.sum()} val={va.sum()} "
        f"test={te.sum()} ratio={ratio:.3f}")

    scaler = StandardScaler().fit(data.X_raw[tr])
    X = scaler.transform(data.X_raw).astype(np.float32)
    NBR = neighbor_aggregate_features(X[:, :93], data.edge_index, len(X))
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    # Subgraph of steps 1-34 for GAT training (no edge crosses steps, so nothing is lost).
    sub_mask = data.time_steps <= 34
    sub_idx, _ = induced_subgraph(sub_mask, data.edge_index)

    def gat_edges(direction, mask=None):
        if mask is None:
            return make_edge_index(data.edge_index, direction)
        _, e = induced_subgraph(mask, data.edge_index)
        return make_edge_index(e, direction)

    quick = args.quick
    max_epochs = 6 if quick else args.max_epochs
    patience = 3 if quick else args.patience

    # ------------------------------------------------------------------ Stage A
    grid_path = os.path.join(args.out, "grid_search.json")
    if os.path.exists(grid_path) and not quick:
        grid = json.load(open(grid_path))
        log("reusing", grid_path)
    else:
        grid = {"hgb": [], "xgb": [], "gat": []}
        hgb_grid = [dict(learning_rate=lr, max_leaf_nodes=ml, l2_regularization=1.0, max_iter=200, early_stopping=False)
                    for lr in (0.05, 0.1) for ml in (15, 31, 63)]
        xgb_grid = [dict(n_estimators=300, max_depth=d, learning_rate=lr, subsample=0.8, colsample_bytree=0.8)
                    for d in (4, 6, 8) for lr in (0.05, 0.1)]
        if quick:
            hgb_grid, xgb_grid = hgb_grid[:2], xgb_grid[:2]
        for p in hgb_grid:
            _, (pv,), s = fit_predict(hgb(p, 42, ratio), X[tr], y_tr, [X[va]])
            grid["hgb"].append({"params": p, "val_ap": float(average_precision_score(y_va, pv)), "fit_s": s})
        for p in xgb_grid:
            _, (pv,), s = fit_predict(xgbc(p, 42, ratio), X[tr], y_tr, [X[va]])
            grid["xgb"].append({"params": p, "val_ap": float(average_precision_score(y_va, pv)), "fit_s": s})
        log("tabular grids done")
        Xs, ys = X[sub_idx], y[sub_idx]
        trs, vas = tr[sub_idx], va[sub_idx]
        gat_grid = [dict(direction=d, hidden=h, heads=4, dropout=0.3, lr=0.005) for d in ("forward", "bidirectional")
                    for h in (32, 64)]
        if quick:
            gat_grid = gat_grid[:1]

        def run_gat(cfg):
            e = gat_edges(cfg["direction"], sub_mask)
            kw = {k: v for k, v in cfg.items() if k != "direction"}
            _, info = train_gat(Xs, e, ys, trs, vas, seed=42, class_weight=ratio, max_epochs=max_epochs,
                                patience=patience, **kw)
            rec = {"params": cfg, "val_ap": info["best_val_ap"], "best_epoch": info["best_epoch"],
                   "train_s": info["train_seconds"]}
            grid["gat"].append(rec)
            log("gat", cfg, round(rec["val_ap"], 4), rec["best_epoch"])

        for cfg in gat_grid:
            run_gat(cfg)
        if not quick:
            best = max(grid["gat"], key=lambda r: r["val_ap"])["params"]
            for change in (dict(dropout=0.5), dict(heads=8), dict(lr=0.01), dict(class_weight_off=True)):
                cfg = {**best, **change}
                if cfg.pop("class_weight_off", False):
                    e = gat_edges(cfg["direction"], sub_mask)
                    kw = {k: v for k, v in cfg.items() if k != "direction"}
                    _, info = train_gat(Xs, e, ys, trs, vas, seed=42, class_weight=None, max_epochs=max_epochs,
                                        patience=patience, **kw)
                    grid["gat"].append({"params": {**cfg, "class_weight": "none"}, "val_ap": info["best_val_ap"],
                                        "best_epoch": info["best_epoch"], "train_s": info["train_seconds"]})
                    log("gat unweighted", round(info["best_val_ap"], 4))
                else:
                    run_gat(cfg)
        if not quick:
            json.dump(grid, open(grid_path, "w"), indent=2)
    best_hgb = max(grid["hgb"], key=lambda r: r["val_ap"])["params"]
    best_xgb = max(grid["xgb"], key=lambda r: r["val_ap"])["params"]
    weighted_gat = [r for r in grid["gat"] if r["params"].get("class_weight") != "none"]
    best_gat = dict(max(weighted_gat, key=lambda r: r["val_ap"])["params"])
    log("selected hgb", best_hgb, "xgb", best_xgb, "gat", best_gat)
    gat_dir = best_gat.pop("direction")
    gat_kw = dict(best_gat, class_weight=ratio, max_epochs=max_epochs, patience=patience)

    legacy_hgb = dict(max_iter=100)
    legacy_xgb = dict(n_estimators=100, max_depth=6, learning_rate=0.1)
    rf_params = dict(n_estimators=200, max_features=50, min_samples_leaf=1, n_jobs=8)

    # ------------------------------------------------------------------ Stage B
    records, prob_frames, fitted_42 = [], [], {}
    test_ids, test_steps = data.tx_ids[te], data.time_steps[te]
    fitted_dir = os.path.join(args.out, "fitted_seed42")
    for seed in ([] if args.aggregate_only else args.seeds):
        seed_file = os.path.join(args.out, f"seed_{seed}.json")
        log(f"===== seed {seed}")
        val_p, test_p, fit_s = {}, {}, {}

        def tab(name, model, cols=None, extra=None):
            Xa = X if cols is None else X[:, cols]
            if extra is not None:
                Xa = np.hstack([Xa, extra])
            m, (pv, pt), s = fit_predict(model, Xa[tr], y_tr, [Xa[va], Xa[te]])
            val_p[name], test_p[name], fit_s[name] = pv, pt, s
            if seed == 42:
                fitted_42[name] = m

        tab("hgb_165_legacy", hgb(legacy_hgb, seed, ratio))
        tab("hgb_165_tuned", hgb(best_hgb, seed, ratio))
        tab("xgb_165_legacy", xgbc(legacy_xgb, seed, ratio))
        tab("xgb_165_tuned", xgbc(best_xgb, seed, ratio))
        tab("rf_165", RandomForestClassifier(class_weight={0: 1.0, 1: ratio}, random_state=seed, **rf_params))
        tab("hgb_lf93_tuned", hgb(best_hgb, seed, ratio), cols=np.arange(93))
        tab("hgb_165_nbr", hgb(best_hgb, seed, ratio), extra=NBR)
        log("tabular done")

        # GAT on the steps 1-34 subgraph; embeddings for every node from the full graph.
        Xs, ys, trs, vas = X[sub_idx], y[sub_idx], tr[sub_idx], va[sub_idx]
        e_sub = gat_edges(gat_dir, sub_mask)
        gat, info = train_gat(Xs, e_sub, ys, trs, vas, seed=seed, **gat_kw)
        p_all, emb_all = gat.predict(X, gat_edges(gat_dir))
        val_p["gat_alone"], test_p["gat_alone"], fit_s["gat_alone"] = p_all[va], p_all[te], info["train_seconds"]
        log("gat", info["best_epoch"], round(info["best_val_ap"], 4), round(info["train_seconds"], 1))
        t0 = time.time()
        emb_cf_sub = cross_fitted_train_embeddings(Xs, e_sub, ys, data.time_steps[sub_idx], trs, vas, seed, gat_kw,
                                                   n_folds=3)
        cf_s = time.time() - t0
        emb_cf = emb_all.copy()
        emb_cf[sub_idx[trs]] = emb_cf_sub[trs]  # only training rows are replaced
        log("cross-fit embeddings", round(cf_s, 1))

        def fused(name, model, emb):
            Xa = np.hstack([X, emb])
            m, (pv, pt), s = fit_predict(model, Xa[tr], y_tr, [Xa[va], Xa[te]])
            val_p[name], test_p[name] = pv, pt
            fit_s[name] = s + info["train_seconds"] + (cf_s if emb is emb_cf else 0.0)
            if seed == 42:
                fitted_42[name] = m

        fused("xgb_165_gat_cf", xgbc(best_xgb, seed, ratio), emb_cf)
        fused("hgb_165_gat_cf", hgb(best_hgb, seed, ratio), emb_cf)
        fused("hgb_165_gat_insample", hgb(best_hgb, seed, ratio), emb_all)
        val_p["ens_hgb_hgbgat"] = 0.5 * (val_p["hgb_165_tuned"] + val_p["hgb_165_gat_cf"])
        test_p["ens_hgb_hgbgat"] = 0.5 * (test_p["hgb_165_tuned"] + test_p["hgb_165_gat_cf"])
        fit_s["ens_hgb_hgbgat"] = fit_s["hgb_165_tuned"] + fit_s["hgb_165_gat_cf"]
        if seed == 42:
            fitted_42["gat"] = gat
            fitted_42["gat_history"] = info["history"]

        for name in val_p:
            th, vf1 = tune_threshold(y_va, val_p[name])          # validation only
            res = evaluate(y_te, test_p[name], th)               # test read once, threshold fixed
            records.append({"seed": seed, "config": name, "val_f1": vf1,
                            "val_ap": float(average_precision_score(y_va, val_p[name])),
                            "fit_seconds": fit_s[name], **res})
            prob_frames.append(pd.DataFrame({"seed": seed, "config": name, "split": "test", "tx_id": test_ids,
                                             "time_step": test_steps, "y_true": y_te, "prob": test_p[name]}))
            prob_frames.append(pd.DataFrame({"seed": seed, "config": name, "split": "val", "tx_id": data.tx_ids[va],
                                             "time_step": data.time_steps[va], "y_true": y_va, "prob": val_p[name]}))
        json.dump([r for r in records if r["seed"] == seed], open(seed_file, "w"), indent=1)
        pd.concat([f for f in prob_frames if f["seed"].iat[0] == seed]).to_parquet(
            os.path.join(args.out, f"seed_{seed}_probs.parquet"), index=False)
        if seed == 42 and not quick:  # persist fitted seed-42 models so a separate aggregate run can export
            os.makedirs(fitted_dir, exist_ok=True)
            torch.save({"config": fitted_42["gat"].config, "state_dict": fitted_42["gat"].state_dict()},
                       os.path.join(fitted_dir, "gat.pt"))
            joblib.dump({k: v for k, v in fitted_42.items() if k not in ("gat", "gat_history")},
                        os.path.join(fitted_dir, "models.joblib"))
        best_line = sorted([r for r in records if r["seed"] == seed], key=lambda r: -r["f1_illicit"])[:4]
        log("top test F1 (info only):", [(r["config"], round(r["f1_illicit"], 4)) for r in best_line])

    if args.train_only:
        log("train-only: per-seed outputs written for", args.seeds)
        return
    if args.aggregate_only:
        for seed in args.seeds:
            records.extend(json.load(open(os.path.join(args.out, f"seed_{seed}.json"))))
            prob_frames.append(pd.read_parquet(os.path.join(args.out, f"seed_{seed}_probs.parquet")))
        if 42 in args.seeds:
            from src.models.graph_encoder import load_encoder
            fitted_42 = joblib.load(os.path.join(fitted_dir, "models.joblib"))
            fitted_42["gat"] = load_encoder(torch.load(os.path.join(fitted_dir, "gat.pt"), weights_only=True))
        log(f"aggregating {len(records)} records from seeds {args.seeds}")
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out, "seed_results.csv"), index=False)
    pd.concat(prob_frames).to_parquet(os.path.join(args.out, "predictions_all_seeds.parquet"), index=False)

    agg = df.groupby("config").agg(
        **{f"{m}_{s}": (m, s) for m in ("val_f1", "val_ap", "precision", "recall", "f1_illicit", "roc_auc",
                                        "average_precision", "threshold", "fp", "fn", "fit_seconds")
           for s in ("mean", "std")}).reset_index()
    agg["components"] = agg["config"].map(COMPONENTS)

    # Selection on validation only.
    top = agg["val_f1_mean"].max()
    cands = agg[agg["val_f1_mean"] >= top - 0.005].sort_values(["val_ap_mean", "components"], ascending=[False, True])
    selected = cands.iloc[0]["config"]
    agg.sort_values("val_f1_mean", ascending=False).to_csv(os.path.join(args.out, "summary.csv"), index=False)

    # Per-time-step test metrics (mean over seeds) for every config.
    preds = pd.concat(prob_frames)
    step_rows = []
    for (cfg, seed), g in preds[preds.split == "test"].groupby(["config", "seed"]):
        th = df[(df.config == cfg) & (df.seed == seed)]["threshold"].iloc[0]
        for t, gs in g.groupby("time_step"):
            pred = (gs.prob >= th).astype(int)
            tp = int(((pred == 1) & (gs.y_true == 1)).sum()); fp = int(((pred == 1) & (gs.y_true == 0)).sum())
            fn = int(((pred == 0) & (gs.y_true == 1)).sum())
            step_rows.append({"config": cfg, "seed": seed, "time_step": t, "n_labelled": len(gs),
                              "n_illicit": int(gs.y_true.sum()), "tp": tp, "fp": fp, "fn": fn,
                              "precision": tp / (tp + fp) if tp + fp else np.nan,
                              "recall": tp / (tp + fn) if tp + fn else np.nan,
                              "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan})
    pd.DataFrame(step_rows).to_csv(os.path.join(args.out, "per_step_metrics.csv"), index=False)

    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.source, "seeds": args.seeds, "quick": quick,
        "splits": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                   "unknown": int((y == -1).sum()), "train_illicit": int(y_tr.sum()),
                   "val_illicit": int(y_va.sum()), "test_illicit": int(y_te.sum())},
        "class_weight_ratio": ratio,
        "selected_hyperparameters": {"hgb": best_hgb, "xgb": best_xgb, "gat": {**best_gat, "direction": gat_dir}},
        "selection_rule": "max mean validation F1 over seeds; within 0.005 -> max mean validation AP, then fewer components",
        "selected_config": selected,
    }
    json.dump(meta, open(os.path.join(args.out, "selection.json"), "w"), indent=2)
    log("SELECTED (validation rule):", selected)
    print(agg.sort_values("val_f1_mean", ascending=False)[["config", "val_f1_mean", "val_ap_mean", "f1_illicit_mean",
                                                            "f1_illicit_std", "precision_mean", "recall_mean",
                                                            "roc_auc_mean", "average_precision_mean"]].round(4).to_string())

    if quick:
        return
    # ------------------------------------------------------------------ export
    from src.models.elliptic_bundle import export_bundle
    r42 = df[(df.config == selected) & (df.seed == 42)].iloc[0].to_dict()
    path = export_bundle(
        out_root=args.artifact_dir, version=args.model_version, config=selected, fitted=fitted_42,
        scaler=scaler, gat_direction=gat_dir, threshold=float(r42["threshold"]),
        metrics_seed42=r42, summary=agg[agg.config == selected].iloc[0].to_dict(), meta=meta)
    log("exported model bundle to", path)


if __name__ == "__main__":
    main()
