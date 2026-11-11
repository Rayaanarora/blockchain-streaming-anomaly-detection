"""
Export the validation-tuned TGAT-BoostNet (notebook 04) as served Elliptic bundle 4.0.0, from the artefacts of
one Kaggle GPU run (scripts/kaggle_train_tgat_boostnet.py). No training happens here.

Model (src/evaluation/tgat_boostnet.py, configuration from results/tuning/selected.json):
    165 standardised features --(2-layer GAT, 8 heads, per time-step snapshot)--> 32-d embedding
    [165 || 32] = 197 features --> XGBoost (500 trees, depth 8, lr 0.03, w_illicit 4.32) --> P(illicit)
    decision threshold 0.37 = validation-F1 maximiser of the tuning stage (fit 1-29, validate 30-34)
Train 1-34, test 35-49, illicit positive, seed 42 (first protocol seed, fixed before export).
Standardisation: FixedStandardiser with the shipped float64 training mean/std (same arithmetic as the
evaluated runs' _standardise_on), not sklearn's StandardScaler.

Checks (all computed and written to <out-dir>/reproduction.json, pass or fail; gates fixed in advance below):
  C0 integrity      file sha256 vs run_meta; local data array hashes vs Kaggle's; local mu/sd vs shipped;
                    sha256 of the local standardised test features vs Kaggle's (exact)
  C1 GPU vs GPU     Kaggle p_test_gpu vs the evaluated Colab run results/runs/tgat_boostnet_tuned/seed_42.npz
  C2 portability    local CPU booster (.ubj) on [local X, Kaggle emb_test_gpu] vs Kaggle p_test_gpu
  C3 CPU vs CPU     local GAT + booster on CPU vs Kaggle's CPU replay (report only)
  C4 device effect  local CPU scores vs Kaggle p_test_gpu
  -> stop (no export) if C0, C1, C2 or C4 fails; --dry-run stops here
  export to artifacts/models/_staging_elliptic/<version>
  C5 bundle         reload the staged bundle, score the 15 test steps as complete snapshots from the Spark output,
                    compare with the local CPU scores; non-gating node/edge order diagnostic on two steps
  -> only if C5 passes: move staging to artifacts/models/elliptic/<version> and write bundle_checks.json

The bundle's reported test metrics are those of the LOCAL CPU model (what is served); the Kaggle GPU and Colab
GPU numbers go into training_meta, labelled by device.

    .venv\\Scripts\\python.exe scripts\\export_tgat_boostnet.py --dry-run
    .venv\\Scripts\\python.exe scripts\\export_tgat_boostnet.py                     # -> artifacts/models/elliptic/4.0.0
Artefacts produced with --epochs (run_meta "smoke_test": true) are only accepted with --dry-run.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.evaluation.data import SEEDS, TEST_STEPS, TRAIN_STEPS, bidirectional, load_elliptic  # noqa: E402
from src.evaluation.tgat_boostnet import _standardise_on, gat_embed  # noqa: E402
from src.models.elliptic_bundle import (FEATURE_COLUMNS, EllipticModelBundle, FixedStandardiser,  # noqa: E402
                                        export_bundle)
from src.models.elliptic_training import evaluate  # noqa: E402
from src.models.graph_encoder import load_encoder  # noqa: E402

DEFAULT_ARTIFACTS = os.path.join(ROOT, "data/validation/tgat_boostnet_bundle/kaggle_seed42")
DEFAULT_OUT = os.path.join(ROOT, "data/validation/tgat_boostnet_bundle")
SEL = os.path.join(ROOT, "results/tuning/selected.json")
RUN = os.path.join(ROOT, "results/runs/tgat_boostnet_tuned")
STAGING_ROOT = os.path.join(ROOT, "artifacts/models/_staging_elliptic")
FINAL_ROOT = os.path.join(ROOT, "artifacts/models/elliptic")
SEED = 42

# ---- gates, fixed before any Kaggle result exists (do not loosen) ----
C1_F1_TOL = 0.02          # |F1(Kaggle GPU) - F1(Colab GPU)|
C1_AP_TOL = 0.02          # |PR-AUC(Kaggle GPU) - PR-AUC(Colab GPU)|
C2_MAX_ABS = 1e-5         # local CPU booster on Kaggle's GPU embeddings vs Kaggle's GPU scores
C2_MAX_FLIPS = 0
C4_MIN_AGREEMENT = 0.999  # local CPU (served) vs Kaggle GPU, decisions at the threshold
C4_F1_TOL = 0.005
C4_AP_TOL = 0.005
C5_MAX_ABS = 1e-5         # staged bundle (per-snapshot scoring) vs local CPU scores; F1 must be identical
ORDER_STEPS = (35, 42)    # non-gating order diagnostic
ORDER_TOL = 1e-6


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rel(path):
    """Repo-relative path with forward slashes, or the absolute path if outside the repo (other drive)."""
    try:
        r = os.path.relpath(path, ROOT)
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")
    return os.path.abspath(path).replace("\\", "/") if r.startswith("..") else r.replace("\\", "/")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def compare(p, q, y, thr):
    """Numbers for p vs q (q is the reference)."""
    d = np.abs(p.astype(np.float64) - q.astype(np.float64))
    mp, mq = evaluate(y, p, thr), evaluate(y, q, thr)
    return {"metrics": mp, "metrics_reference": mq,
            "f1_diff": mp["f1_illicit"] - mq["f1_illicit"],
            "pr_auc_diff": mp["average_precision"] - mq["average_precision"],
            "max_abs_diff": float(d.max()), "median_abs_diff": float(np.median(d)),
            "p99_abs_diff": float(np.quantile(d, 0.99)),
            "n_above_1e-4": int((d > 1e-4).sum()), "n_above_1e-3": int((d > 1e-3).sum()),
            "n_above_1e-2": int((d > 1e-2).sum()),
            "decision_flips": int(((p >= thr) != (q >= thr)).sum()),
            "decision_agreement": float(((p >= thr) == (q >= thr)).mean())}


def load_gat(art, run_meta):
    try:
        g = load_encoder(torch.load(os.path.join(art, "gat.pt"), weights_only=True, map_location="cpu"))
        return g, "gat.pt"
    except Exception as e:  # noqa: BLE001
        log(f"torch.load(gat.pt) failed ({type(e).__name__}: {str(e)[:120]}); using gat_weights.npz")
        z = np.load(os.path.join(art, "gat_weights.npz"))
        state = {"config": run_meta["gat_config"], "state_dict": {k: torch.from_numpy(z[k].copy()) for k in z.files}}
        return load_encoder(state), "gat_weights.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default=DEFAULT_ARTIFACTS, help="unzipped Kaggle output folder")
    ap.add_argument("--version", default="4.0.0")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="where reproduction.json / bundle_checks.json go")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="checks C0-C4 only; nothing is exported")
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    art = os.path.abspath(args.artifacts)
    out = os.path.abspath(args.out_dir)
    if not os.path.isfile(os.path.join(art, "run_meta.json")):
        raise SystemExit(f"{art}/run_meta.json not found (unzip the Kaggle zip into --artifacts)")
    run_meta = json.load(open(os.path.join(art, "run_meta.json")))
    smoke = bool(run_meta.get("smoke_test"))
    if smoke and not args.dry_run:
        raise SystemExit("these artefacts are a smoke test (run_meta smoke_test=true); only --dry-run is allowed")
    final_dir = os.path.join(FINAL_ROOT, args.version)
    staging_dir = os.path.join(STAGING_ROOT, args.version)
    if not args.dry_run:
        if os.path.exists(final_dir):
            raise SystemExit(f"{final_dir} already exists; refusing to overwrite (bump --version)")
        if os.path.exists(staging_dir):
            raise SystemExit(f"stale staging directory {staging_dir}; inspect and remove it first")
    os.makedirs(out, exist_ok=True)

    sel = json.load(open(SEL))["tgat_boostnet"]
    threshold = float(sel["threshold"])
    if float(run_meta["threshold"]) != threshold or int(run_meta["seed"]) != SEED:
        raise SystemExit(f"run_meta threshold/seed {run_meta['threshold']}/{run_meta['seed']} != {threshold}/{SEED}")
    log(f"artefacts {art} (smoke_test={smoke}); threshold {threshold}; seed {SEED}")

    # ------------------------------------------------------------- load
    data = load_elliptic(verbose=False)
    tr, te = data.train_mask, data.test_mask
    y_te = data.y[te]
    to = np.load(os.path.join(art, "test_outputs.npz"))
    ss = np.load(os.path.join(art, "scaler_stats.npz"))
    mu_k, sd_k = ss["mu"], ss["sd"]
    p_gpu, emb_gpu = to["p_test_gpu"], to["emb_test_gpu"]
    gat, gat_source = load_gat(art, run_meta)
    import xgboost as xgb
    booster = xgb.XGBClassifier()
    booster.load_model(os.path.join(art, "tgat_boostnet_xgb.ubj"))

    # ------------------------------------------------------------- C0 integrity
    files = {k: {"expected": h, "local": sha256_file(os.path.join(art, k)) if os.path.exists(os.path.join(art, k)) else None}
             for k, h in run_meta["files_sha256"].items()}
    files_ok = all(v["expected"] == v["local"] for v in files.values())
    data_local = {"X_raw": data.X_raw, "tx_ids": data.tx_ids, "y": data.y, "edge_index": data.edge_index,
                  "time_steps": data.time_steps}
    data_h = {k: {"kaggle": run_meta["data_sha256"].get(k), "local": sha256_array(a)} for k, a in data_local.items()}
    data_ok = all(v["kaggle"] == v["local"] for v in data_h.values())
    rows_ok = bool(np.array_equal(to["tx_ids"], data.tx_ids[te]) and np.array_equal(to["y"], y_te)
                   and np.array_equal(to["time_step"], data.time_steps[te]))
    fit_rows = data.mask(TRAIN_STEPS)
    mu_l = data.X_raw[fit_rows].astype(np.float64).mean(0)
    sd_l = data.X_raw[fit_rows].astype(np.float64).std(0)
    sd_l[sd_l == 0] = 1.0
    scaler = FixedStandardiser(mu_k, sd_k)
    X = scaler.transform(data.X_raw).astype(np.float32)          # exactly what the bundle computes
    X_eval = _standardise_on(data.X_raw, fit_rows)                # what the evaluated runs computed locally
    feat_k = str(ss["x_test_std_sha256"])
    feat_served = sha256_array(X[te])
    c0 = {"files": files, "files_ok": files_ok, "data": data_h, "data_ok": data_ok, "test_rows_aligned": rows_ok,
          "mu_bit_equal": bool(np.array_equal(mu_l, mu_k)), "sd_bit_equal": bool(np.array_equal(sd_l, sd_k)),
          "mu_max_abs_diff": float(np.max(np.abs(mu_l - mu_k))), "sd_max_abs_diff": float(np.max(np.abs(sd_l - sd_k))),
          "x_test_sha256_kaggle": feat_k, "x_test_sha256_local_fixed_standardiser": feat_served,
          "x_test_sha256_local_standardise_on": sha256_array(X_eval[te]),
          "fixed_standardiser_equals_standardise_on": bool(np.array_equal(X, X_eval)),
          "gat_loaded_from": gat_source}
    c0["feature_hash_ok"] = feat_served == feat_k
    c0["pass"] = bool(files_ok and data_ok and rows_ok and c0["feature_hash_ok"])
    del X_eval
    log(f"C0 integrity: files {files_ok}, data {data_ok}, rows {rows_ok}, mu/sd bit-equal "
        f"{c0['mu_bit_equal']}/{c0['sd_bit_equal']}, feature hash {c0['feature_hash_ok']} -> pass={c0['pass']}")

    # ------------------------------------------------------------- local CPU scores (served model)
    E = bidirectional(data.edge_index)
    test_steps = (data.time_steps >= TEST_STEPS[0]) & (data.time_steps <= TEST_STEPS[1])
    t0 = time.time()
    idx, _, emb = gat_embed(gat, X, E, test_steps, "cpu")
    emb_all = np.zeros((data.n, emb.shape[1]), dtype=np.float32)
    emb_all[idx] = emb
    emb_local = emb_all[te]
    del emb_all, emb
    p_local = booster.predict_proba(np.hstack([X[te], emb_local]))[:, 1].astype(np.float32)
    p_c2 = booster.predict_proba(np.hstack([X[te], emb_gpu]))[:, 1].astype(np.float32)
    local_s = time.time() - t0

    # ------------------------------------------------------------- C1 Kaggle GPU vs Colab GPU
    p_colab = np.load(os.path.join(RUN, f"seed_{SEED}.npz"))["p_test"].astype(np.float32)
    c1 = compare(p_gpu, p_colab, y_te, threshold)
    seed_f1 = {}
    for s in SEEDS:
        m = json.load(open(os.path.join(RUN, f"seed_{s}.json")))
        seed_f1[s] = evaluate(y_te, np.load(os.path.join(RUN, f"seed_{s}.npz"))["p_test"], float(m["threshold"]))["f1_illicit"]
    lo, hi = min(seed_f1.values()), max(seed_f1.values())
    c1.update({"reference": f"results/runs/tgat_boostnet_tuned/seed_{SEED}.npz (Colab T4 GPU)",
               "five_seed_f1": seed_f1, "five_seed_f1_range": [lo, hi],
               "kaggle_f1_inside_five_seed_range": bool(lo <= c1["metrics"]["f1_illicit"] <= hi),
               "gate": {"abs_f1_diff_max": C1_F1_TOL, "abs_pr_auc_diff_max": C1_AP_TOL}})
    c1["pass"] = bool(abs(c1["f1_diff"]) <= C1_F1_TOL and abs(c1["pr_auc_diff"]) <= C1_AP_TOL)
    log(f"C1 GPU vs GPU: dF1 {c1['f1_diff']:+.4f}, dPR-AUC {c1['pr_auc_diff']:+.4f} -> pass={c1['pass']}")

    # ------------------------------------------------------------- C2 portability
    c2 = compare(p_c2, p_gpu, y_te, threshold)
    c2["vs_kaggle_cpu_booster_on_gpu_emb_max_abs_diff"] = float(np.max(np.abs(p_c2 - to["p_xgbcpu_embgpu"])))
    c2["gate"] = {"max_abs_diff": C2_MAX_ABS, "max_decision_flips": C2_MAX_FLIPS}
    c2["pass"] = bool(c2["max_abs_diff"] <= C2_MAX_ABS and c2["decision_flips"] <= C2_MAX_FLIPS)
    log(f"C2 portability: max diff {c2['max_abs_diff']:.2e}, flips {c2['decision_flips']} -> pass={c2['pass']}")

    # ------------------------------------------------------------- C3 CPU vs CPU (report only)
    c3 = compare(p_local, to["p_test_cpu"], y_te, threshold)
    de = np.abs(emb_local.astype(np.float64) - to["emb_test_cpu"].astype(np.float64))
    c3.update({"emb_max_abs_diff": float(de.max()), "emb_median_abs_diff": float(np.median(de)),
               "emb_bit_equal": bool(np.array_equal(emb_local, to["emb_test_cpu"])),
               "emb_local_vs_kaggle_gpu_max_abs_diff": float(np.max(np.abs(emb_local - emb_gpu))),
               "gating": False})
    log(f"C3 CPU vs CPU (report): emb max diff {c3['emb_max_abs_diff']:.2e}, score max diff {c3['max_abs_diff']:.2e}")

    # ------------------------------------------------------------- C4 device effect (served CPU vs Kaggle GPU)
    c4 = compare(p_local, p_gpu, y_te, threshold)
    flips = np.flatnonzero((p_local >= threshold) != (p_gpu >= threshold))
    c4["flipped_rows"] = [{"tx_id": int(to["tx_ids"][i]), "time_step": int(to["time_step"][i]), "label": int(y_te[i]),
                           "p_local_cpu": float(p_local[i]), "p_kaggle_gpu": float(p_gpu[i])} for i in flips]
    c4["gate"] = {"min_decision_agreement": C4_MIN_AGREEMENT, "abs_f1_diff_max": C4_F1_TOL, "abs_pr_auc_diff_max": C4_AP_TOL}
    c4["pass"] = bool(c4["decision_agreement"] >= C4_MIN_AGREEMENT and abs(c4["f1_diff"]) <= C4_F1_TOL
                      and abs(c4["pr_auc_diff"]) <= C4_AP_TOL)
    log(f"C4 device effect: agreement {c4['decision_agreement']:.5f}, dF1 {c4['f1_diff']:+.4f}, "
        f"dPR-AUC {c4['pr_auc_diff']:+.4f}, flips {c4['decision_flips']} -> pass={c4['pass']}")

    repro = {"artifacts": art, "smoke_test": smoke,
             "smoke_test_note": "PLUMBING CHECK ONLY: artefacts from a shortened run; numbers are not model results"
             if smoke else None,
             "seed": SEED, "threshold": threshold, "local_cpu_seconds": local_s,
             "kaggle_versions": run_meta.get("versions"), "kaggle_repo": run_meta.get("repo"),
             "C0_integrity": c0, "C1_gpu_vs_gpu": c1, "C2_portability": c2, "C3_cpu_vs_cpu": c3, "C4_device_effect": c4}
    gated = {"C0": c0["pass"], "C1": c1["pass"], "C2": c2["pass"], "C4": c4["pass"]}
    repro["gates_passed"] = gated
    repro["export_allowed"] = all(gated.values()) and not smoke
    rpath = os.path.join(out, "reproduction.json")
    dump(repro, rpath)
    pd.DataFrame({"tx_id": to["tx_ids"], "time_step": to["time_step"], "label": y_te, "score_local_cpu": p_local,
                  "score_kaggle_gpu": p_gpu, "score_kaggle_cpu": to["p_test_cpu"], "score_colab_gpu": p_colab}
                 ).to_parquet(os.path.join(out, "test_scores.parquet"), index=False)
    log("wrote", rpath)
    failed = [k for k, v in gated.items() if not v]
    if failed:
        raise SystemExit(f"checks failed: {failed}; NOT exporting (see {rpath})")
    if args.dry_run:
        log("dry run: C0-C4 passed, stopping before export")
        return

    # ------------------------------------------------------------- export to staging
    summ = pd.read_csv(os.path.join(ROOT, "tables/04_tuned_vs_untuned.csv"), index_col=0)
    row = summ.loc["TGAT-BoostNet (ours) (tuned)"].to_dict()
    cfg = run_meta["config"]
    blocks = np.array_split(np.arange(TRAIN_STEPS[0], TRAIN_STEPS[1] + 1), cfg["n_folds"])
    metrics = dict(evaluate(y_te, p_local, threshold),
                   device="local CPU (served model: Kaggle-trained weights, scored on CPU)",
                   note="seed 42 at the validation threshold; test 35-49 scored once")
    meta = {"model_name": "TGAT-BoostNet", "source": "src/evaluation/tgat_boostnet.py; notebook 04 (tuned)",
            "selected_configuration": sel, "export_seed_rule": "seed 42 (first protocol seed), fixed before export",
            "training": "scripts/kaggle_train_tgat_boostnet.py, one Kaggle run (first complete run exported)",
            "training_device": run_meta.get("device"), "training_gpu": (run_meta.get("versions") or {}).get("gpu"),
            "training_versions": run_meta.get("versions"), "training_repo": run_meta.get("repo"),
            "training_timings_seconds": run_meta.get("timings_seconds"),
            "artifact_sha256": run_meta["files_sha256"], "gat": gat.config, "gat_loaded_from": gat_source,
            "gat_epochs": cfg["epochs"], "cross_fit_blocks": [[int(b[0]), int(b[-1])] for b in blocks],
            "train_steps": list(TRAIN_STEPS), "test_steps": list(TEST_STEPS),
            "standardiser": "FixedStandardiser (float64 training mean/std of labelled steps 1-34)",
            "test_metrics_by_device": {"local_cpu_served": c4["metrics"],
                                       "kaggle_gpu": c1["metrics"], "kaggle_cpu_replay": c3["metrics_reference"],
                                       "colab_gpu_evaluated_run": c1["metrics_reference"]},
            "reproduction": rel(rpath),
            "five_seed_source": "tables/04_tuned_vs_untuned.csv (evaluated runs, Colab T4)"}
    os.makedirs(STAGING_ROOT, exist_ok=True)
    path = export_bundle(
        STAGING_ROOT, args.version, "tgat_boostnet", {"tgat_boostnet_xgb": booster, "gat": gat}, scaler,
        "bidirectional", threshold, metrics, row, meta, seed=SEED,
        score_semantics="Probability-like score of the Elliptic 'illicit' class (label 1) from TGAT-BoostNet "
                        "(GAT embedding + XGBoost) trained on labelled steps 1-34. Not calibrated across datasets.",
        threshold_source="validation-F1 maximiser of the tuning stage (fit 1-29, validate 30-34); "
                         "results/tuning/selected.json")
    log("staged", path)

    # ------------------------------------------------------------- C5 staged bundle, per-snapshot scoring
    b = EllipticModelBundle(path)
    nodes = pd.read_parquet(os.path.join(ROOT, "data/processed/elliptic_spark/nodes"),
                            columns=["tx_id", "time_step"] + FEATURE_COLUMNS, filters=[("time_step", ">=", TEST_STEPS[0])])
    nodes["time_step"] = nodes["time_step"].astype(int)
    edges = pd.read_parquet(os.path.join(ROOT, "data/processed/elliptic_spark/edges"),
                            columns=["source_tx_id", "target_tx_id", "time_step"],
                            filters=[("time_step", ">=", TEST_STEPS[0])])
    got, contexts, t0 = [], set(), time.time()
    for step, g in nodes.groupby("time_step"):
        p = b.predict(g[["tx_id"] + FEATURE_COLUMNS], edges[edges.time_step == step])
        contexts |= set(p.graph_context)
        got.append(p)
    snap_s = time.time() - t0
    got = pd.concat(got).set_index("tx_id")
    bundle_scores = got.loc[to["tx_ids"], "score"].to_numpy()
    res_b = evaluate(y_te, bundle_scores, b.threshold)
    res_l = evaluate(y_te, p_local, threshold)
    order = {}
    rng = np.random.default_rng(0)
    for step in ORDER_STEPS:
        g = nodes[nodes.time_step == step]
        e = edges[edges.time_step == step]
        base = b.predict(g[["tx_id"] + FEATURE_COLUMNS], e).set_index("tx_id")["score"]
        sh = b.predict(g.iloc[rng.permutation(len(g))][["tx_id"] + FEATURE_COLUMNS],
                       e.iloc[rng.permutation(len(e))]).set_index("tx_id")["score"]
        d = np.abs(sh.loc[base.index].to_numpy() - base.to_numpy())
        order[int(step)] = {"n_nodes": int(len(g)), "n_changed_gt_1e-6": int((d > ORDER_TOL).sum()),
                            "max_abs_diff": float(d.max()),
                            "decision_flips": int(((sh.loc[base.index].to_numpy() >= b.threshold)
                                                   != (base.to_numpy() >= b.threshold)).sum())}
    diff = float(np.max(np.abs(bundle_scores - p_local)))
    c5 = {"bundle_staged": rel(path), "config": b.manifest["config"],
          "requires_graph": b.requires_graph, "threshold": b.threshold, "graph_contexts": sorted(contexts),
          "snapshots_scored": int(nodes.time_step.nunique()), "nodes_scored": int(len(got)),
          "snapshot_scoring_seconds": snap_s, "max_abs_diff_vs_local_cpu": diff,
          "test_metrics_from_bundle": res_b, "test_metrics_local_cpu": res_l,
          "order_diagnostic_non_gating": order, "gate": {"max_abs_diff": C5_MAX_ABS, "identical_f1": True}}
    c5["pass"] = bool(diff <= C5_MAX_ABS and res_b["f1_illicit"] == res_l["f1_illicit"] and contexts == {"complete"})
    repro["C5_bundle"] = c5
    dump(repro, rpath)
    log(f"C5 bundle: max diff {diff:.2e}, F1 bundle {res_b['f1_illicit']:.4f} vs local {res_l['f1_illicit']:.4f}, "
        f"order diagnostic {order} -> pass={c5['pass']}")
    if not c5["pass"]:
        raise SystemExit(f"C5 failed: staged bundle left in {path}; nothing moved to {FINAL_ROOT} (current bundle stays served)")

    del b
    shutil.move(path, final_dir)
    try:
        os.rmdir(STAGING_ROOT)
    except OSError:
        pass
    EllipticModelBundle(final_dir)  # checksums still valid after the move
    checks = dict(c5, bundle=rel(final_dir), seed=SEED,
                  reproduction=rel(rpath))
    dump(checks, os.path.join(out, "bundle_checks.json"))
    log("exported", final_dir)


if __name__ == "__main__":
    main()
