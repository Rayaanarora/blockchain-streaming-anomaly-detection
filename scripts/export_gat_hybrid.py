"""
Export the deep-learning hybrid (GAT embeddings + gradient boosting) as the served Elliptic model.

Config (default) ``ens_hgb_hgbgat`` = mean of
  (a) HistGBDT on the 165 standardised features, and
  (b) HistGBDT on 165 features + 32 cross-fitted embeddings from a 2-layer, 8-head GAT.
It is the GAT-containing configuration with the highest mean validation F1 in the 5-seed model
selection (data/validation/final_model_selection/summary.csv). The exported copy is the seed with the
highest validation F1 for that config (seed 123); test scores were not used for either choice.

Steps (all must pass before anything is written to artifacts/):
  1. retrain the chosen seed exactly as scripts/run_final_model_selection.py did
  2. reproduction check: every component's validation and test probabilities must match the saved
     data/validation/final_model_selection/seed_<seed>_probs.parquet
  3. export bundle <version>, reload it, score each test time step as one complete snapshot (what the
     streaming consumer does) and check scores and metrics against step 1

    .venv\\Scripts\\python.exe scripts\\export_gat_hybrid.py                 # seed 123 -> bundle 3.0.0
    .venv\\Scripts\\python.exe scripts\\export_gat_hybrid.py --dry-run       # steps 1-2 only

Writes data/validation/gat_hybrid/{reproduction.json,bundle_checks.json} and artifacts/models/elliptic/<version>/.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.elliptic_bundle import EllipticModelBundle, export_bundle  # noqa: E402
from src.models.elliptic_training import (assert_no_cross_step_edges, cross_fitted_train_embeddings,  # noqa: E402
                                          evaluate, induced_subgraph, load_spark_output, train_gat,
                                          tune_threshold)
from src.models.graph_encoder import make_edge_index  # noqa: E402

SEL = os.path.join(ROOT, "data/validation/final_model_selection")
OUT = os.path.join(ROOT, "data/validation/gat_hybrid")
GAT_CONFIGS = ["ens_hgb_hgbgat", "hgb_165_gat_cf", "xgb_165_gat_cf", "hgb_165_gat_insample", "gat_alone"]
PROB_TOL = 1e-6


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="ens_hgb_hgbgat", choices=["ens_hgb_hgbgat", "hgb_165_gat_cf"])
    ap.add_argument("--seed", type=int, default=None, help="default: highest validation F1 for --config")
    ap.add_argument("--version", default="3.0.0")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--max-epochs", type=int, default=150, help="as in the 5-seed run")
    ap.add_argument("--patience", type=int, default=20, help="as in the 5-seed run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    os.makedirs(OUT, exist_ok=True)

    # ------------------------------------------------------------- choices (validation only)
    summary = pd.read_csv(os.path.join(SEL, "summary.csv"))
    dl = summary[summary.config.isin(GAT_CONFIGS)].sort_values("val_f1_mean", ascending=False)
    rule_choice = dl.iloc[0]["config"]
    seeds = pd.read_csv(os.path.join(SEL, "seed_results.csv"))
    per_seed = seeds[seeds.config == args.config].sort_values(["val_f1", "val_ap"], ascending=False)
    seed = args.seed if args.seed is not None else int(per_seed.iloc[0]["seed"])
    log(f"config {args.config} (rule choice among GAT configs: {rule_choice}); seed {seed}")
    sel = json.load(open(os.path.join(SEL, "selection.json")))
    hp = sel["selected_hyperparameters"]
    gat_hp = dict(hp["gat"])
    direction = gat_hp.pop("direction")

    # ------------------------------------------------------------- data, exactly as the 5-seed run
    data = load_spark_output(os.path.join(ROOT, "data/processed/elliptic_spark"))
    assert_no_cross_step_edges(data)
    tr, va, te, y = data.train_mask, data.val_mask, data.test_mask, data.y
    ratio = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    scaler = StandardScaler().fit(data.X_raw[tr])
    X = scaler.transform(data.X_raw).astype(np.float32)
    sub_mask = data.time_steps <= 34
    sub_idx, e_sub0 = induced_subgraph(sub_mask, data.edge_index)
    e_sub = make_edge_index(e_sub0, direction)
    Xs, ys, trs, vas = X[sub_idx], y[sub_idx], tr[sub_idx], va[sub_idx]
    gat_kw = dict(gat_hp, class_weight=ratio, max_epochs=args.max_epochs, patience=args.patience)

    def hgb():
        return HistGradientBoostingClassifier(class_weight={0: 1.0, 1: ratio}, random_state=seed, **hp["hgb"])

    # ------------------------------------------------------------- 1. retrain
    t_all = time.time()
    m_tab = hgb().fit(X[tr], y[tr])
    p_tab = {"val": m_tab.predict_proba(X[va])[:, 1], "test": m_tab.predict_proba(X[te])[:, 1]}
    log("hgb_165_tuned fitted")
    gat, info = train_gat(Xs, e_sub, ys, trs, vas, seed=seed, **gat_kw)
    log(f"GAT: best epoch {info['best_epoch']}, val AP {info['best_val_ap']:.4f}, {info['train_seconds']:.0f}s")
    p_gat_all, emb_all = gat.predict(X, make_edge_index(data.edge_index, direction))
    t0 = time.time()
    emb_cf_sub = cross_fitted_train_embeddings(Xs, e_sub, ys, data.time_steps[sub_idx], trs, vas, seed, gat_kw,
                                               n_folds=3)
    log(f"cross-fitted embeddings: {time.time() - t0:.0f}s")
    emb_cf = emb_all.copy()
    emb_cf[sub_idx[trs]] = emb_cf_sub[trs]
    Xe = np.hstack([X, emb_cf])
    m_fused = hgb().fit(Xe[tr], y[tr])
    p_fused = {"val": m_fused.predict_proba(Xe[va])[:, 1], "test": m_fused.predict_proba(Xe[te])[:, 1]}
    probs = {"hgb_165_tuned": p_tab, "hgb_165_gat_cf": p_fused,
             "gat_alone": {"val": p_gat_all[va], "test": p_gat_all[te]},
             "ens_hgb_hgbgat": {k: 0.5 * (p_tab[k] + p_fused[k]) for k in ("val", "test")}}
    train_seconds = time.time() - t_all

    # ------------------------------------------------------------- 2. reproduction check
    saved = pd.read_parquet(os.path.join(SEL, f"seed_{seed}_probs.parquet"))
    repro = {"seed": seed, "gat_best_epoch": info["best_epoch"], "gat_best_val_ap": info["best_val_ap"],
             "retrain_seconds": round(train_seconds, 1), "torch_threads": args.torch_threads, "components": {}}
    ok = True
    for cfg, pr in probs.items():
        rec = {}
        for split, mask in (("val", va), ("test", te)):
            s = saved[(saved.config == cfg) & (saved.split == split)]
            if not np.array_equal(s.tx_id.to_numpy(), data.tx_ids[mask]):
                raise SystemExit(f"saved {cfg}/{split} rows are not aligned with the data")
            d = float(np.max(np.abs(s.prob.to_numpy() - pr[split])))
            rec[f"{split}_max_abs_prob_diff"] = d
            ok &= d <= PROB_TOL
        repro["components"][cfg] = rec
    th, vf1 = tune_threshold(y[va], probs[args.config]["val"])
    res = evaluate(y[te], probs[args.config]["test"], th)
    rec_saved = [r for r in json.load(open(os.path.join(SEL, f"seed_{seed}.json"))) if r["config"] == args.config][0]
    repro["threshold"] = {"retrained": th, "saved": rec_saved["threshold"]}
    repro["test_f1"] = {"retrained": res["f1_illicit"], "saved": rec_saved["f1_illicit"]}
    repro["reproduced"] = bool(ok and th == rec_saved["threshold"])
    json.dump(repro, open(os.path.join(OUT, "reproduction.json"), "w"), indent=2)
    for cfg, r in repro["components"].items():
        log(f"  {cfg:16s} val diff {r['val_max_abs_prob_diff']:.2e}  test diff {r['test_max_abs_prob_diff']:.2e}")
    log(f"threshold {th} (saved {rec_saved['threshold']}), test F1 {res['f1_illicit']:.4f} "
        f"(saved {rec_saved['f1_illicit']:.4f}) -> reproduced={repro['reproduced']}")
    if not repro["reproduced"]:
        raise SystemExit("retrained model does not reproduce the saved seed run; NOT exporting "
                         f"(see {os.path.join(OUT, 'reproduction.json')})")
    if args.dry_run:
        return

    # ------------------------------------------------------------- 3. export + bundle checks
    row = summary[summary.config == args.config].iloc[0].to_dict()
    meta = {"selection_among_gat_configs": "highest mean validation F1 over 5 seeds among configs containing the "
                                           "GAT (deep learning requirement); test not used",
            "rule_choice": rule_choice, "gat_configs_val_f1_mean": dict(zip(dl.config, dl.val_f1_mean.round(4))),
            "export_seed_rule": "highest validation F1 among the 5 seeds of the chosen config; test not used",
            "per_seed_val_f1": dict(zip(per_seed.seed.astype(int).astype(str), per_seed.val_f1.round(4))),
            "hyperparameters": {"hgb": hp["hgb"], "gat": hp["gat"], "max_epochs": args.max_epochs,
                                "patience": args.patience, "cross_fit_folds": 3},
            "gat_training": {k: info[k] for k in ("best_epoch", "best_val_ap", "epochs_run", "train_seconds")},
            "data_source": "data/processed/elliptic_spark", "reproduction": "data/validation/gat_hybrid/reproduction.json"}
    metrics = {**res, "val_f1": vf1, "val_ap": float(average_precision_score(y[va], probs[args.config]["val"]))}
    path = export_bundle(os.path.join(ROOT, "artifacts/models/elliptic"), args.version, args.config,
                         {"hgb_165_tuned": m_tab, "hgb_165_gat_cf": m_fused, "gat": gat}, scaler, direction,
                         th, metrics, row, meta, seed=seed)
    log("exported", path)

    b = EllipticModelBundle(path)
    nodes = pd.read_parquet(os.path.join(ROOT, "data/processed/elliptic_spark/nodes"), filters=[("time_step", ">=", 35)])
    nodes["time_step"] = nodes["time_step"].astype(int)
    edges = pd.read_parquet(os.path.join(ROOT, "data/processed/elliptic_spark/edges"), filters=[("time_step", ">=", 35)])
    got = []
    t0 = time.time()
    for step, g in nodes.groupby("time_step"):
        p = b.predict(g, edges[edges.time_step == step])
        assert set(p.graph_context) == {"complete"}
        got.append(p)
    snap_s = time.time() - t0
    got = pd.concat(got).set_index("tx_id")
    test_ids = data.tx_ids[te]
    bundle_scores = got.loc[test_ids, "score"].to_numpy()
    diff = float(np.max(np.abs(bundle_scores - probs[args.config]["test"])))
    res_b = evaluate(y[te], bundle_scores, b.threshold)
    checks = {"bundle": path, "config": args.config, "seed": seed, "threshold": b.threshold,
              "requires_graph": b.requires_graph, "snapshots_scored": int(nodes.time_step.nunique()),
              "nodes_scored": int(len(got)), "snapshot_scoring_seconds": round(snap_s, 2),
              "max_abs_diff_vs_training_time_test_probs": diff,
              "test_metrics_from_bundle": res_b, "test_metrics_training_time": res,
              "pass": bool(diff <= 1e-5 and res_b["f1_illicit"] == res["f1_illicit"])}
    json.dump(checks, open(os.path.join(OUT, "bundle_checks.json"), "w"), indent=2, default=float)
    log(f"bundle check: max diff {diff:.2e}, bundle test F1 {res_b['f1_illicit']:.4f} -> pass={checks['pass']}")
    if not checks["pass"]:
        raise SystemExit("exported bundle does not reproduce the evaluated scores")


if __name__ == "__main__":
    main()
