"""
Train the validation-tuned TGAT-BoostNet (seed 42) on a Kaggle T4 GPU and save portable artefacts.

This is the training half of serving TGAT-BoostNet as Elliptic bundle 4.0.0. The bundle itself is built
locally, on CPU, by scripts/export_tgat_boostnet.py from the zip this script writes.

What it does (no hand-rolled training loop: it calls the repo's own functions)
  * configuration from results/tuning/selected.json["tgat_boostnet"], built exactly as
    src/evaluation/tuning.final_run does; training by src.evaluation.tgat_boostnet.fit_tgat_boostnet
    (cross-fitted GAT embeddings for the booster's training rows, GAT on all of steps 1-34 for test rows)
  * train = labelled steps 1-34, test = labelled steps 35-49, illicit is the positive class, seed 42
  * checks the test-set signature against results/runs/tgat_boostnet_tuned/seed_42.json
  * saves only portable files (no pickle / joblib): gat.pt + gat_weights.npz, the booster as .ubj and .json,
    the standardisation statistics, the test outputs on GPU and on CPU, run_meta.json with versions,
    hashes, timings and metrics; then zips the folder to <out>.zip

ONE-RUN RULE: the first complete Kaggle run is the one that is exported. Do not rerun this script to get
better numbers. A rerun is only allowed if the first run did not complete (crash, timeout, lost session),
and that has to be stated with the export.

Kaggle notebook (Settings: Accelerator GPU T4 x1, Internet on). Cells:

    !pip install -q torch_geometric==2.8.1 xgboost==3.4.1      # do NOT reinstall torch (keep Kaggle's CUDA build)

    !git clone https://github.com/Rayaanarora/blockchain-streaming-anomaly-detection.git /kaggle/working/repo
    !git -C /kaggle/working/repo checkout 752a4b53da63d888f4cd30716ed9274a3575aaf1

    # Add the Elliptic dataset as notebook input (Add Input -> "ellipticco/elliptic-data-set"); the CSVs are
    # found under /kaggle/input automatically.
    !cd /kaggle/working && python repo/scripts/kaggle_train_tgat_boostnet.py --repo-dir /kaggle/working/repo

    # then download /kaggle/working/tgat_boostnet_seed42.zip (Output panel) and unzip it locally into
    # data/validation/tgat_boostnet_bundle/kaggle_seed42/

Local versions this must match (the bundle is loaded with them): torch_geometric 2.8.1, xgboost 3.4.1
(torch 2.14.1+cpu, scikit-learn 1.9.1, numpy 2.5.4 locally). An xgboost newer than the local one is refused,
because its model file may not load locally (override: --allow-newer-xgboost).

``--epochs`` exists ONLY for local plumbing smoke tests; a run with it is marked "smoke_test": true in
run_meta.json and scripts/export_tgat_boostnet.py refuses to export it.
"""
import argparse
import copy
import dataclasses
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np

LOCAL_VERSIONS = {"torch": "2.14.1+cpu", "torch_geometric": "2.8.1", "xgboost": "3.4.1",
                  "sklearn": "1.9.1", "numpy": "2.5.4", "python": "3.12.10"}
EXPECTED_REPO_SHA = "752a4b53da63d888f4cd30716ed9274a3575aaf1"
SEED_DEFAULT = 42


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _ver(s: str) -> tuple:
    return tuple(int(p) for p in re.findall(r"\d+", s.split("+")[0])[:3])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def versions() -> dict:
    import sklearn
    import torch
    import torch_geometric
    import xgboost
    v = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
         "torch": torch.__version__, "torch_cuda": torch.version.cuda, "torch_geometric": torch_geometric.__version__,
         "xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
         "cuda_available": bool(torch.cuda.is_available()),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        import pandas
        v["pandas"] = pandas.__version__
    except ImportError:
        pass
    return v


def git_info(repo: str) -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(["git", "-C", repo, *cmd], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:  # noqa: BLE001
            return None
    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"sha": sha, "expected_sha": EXPECTED_REPO_SHA, "matches_expected": sha == EXPECTED_REPO_SHA,
            "tracked_files_modified": None if status is None else bool(status)}


def find_data_dir(arg) -> str:
    if arg:
        return arg
    hits = sorted(glob.glob("/kaggle/input/**/elliptic_txs_features.csv", recursive=True))
    if not hits:
        raise SystemExit("elliptic_txs_features.csv not found under /kaggle/input; add the Elliptic dataset as "
                         "notebook input or pass --data-dir")
    return os.path.dirname(hits[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-dir", required=True, help="checkout of the repository (commit %s)" % EXPECTED_REPO_SHA[:12])
    ap.add_argument("--data-dir", default=None, help="folder with the three Elliptic CSVs "
                                                     "(default: found under /kaggle/input)")
    ap.add_argument("--out", default="/kaggle/working/tgat_boostnet_seed42")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="LOCAL SMOKE TESTS ONLY: override GAT epochs; marks the run smoke_test=true (not exportable)")
    ap.add_argument("--allow-newer-xgboost", action="store_true")
    args = ap.parse_args()
    t_script = time.time()

    repo = os.path.abspath(args.repo_dir)
    sys.path.insert(0, repo)
    v = versions()
    log("versions:", json.dumps(v))
    log("local (bundle-loading) versions:", json.dumps(LOCAL_VERSIONS))
    if _ver(v["xgboost"]) > _ver(LOCAL_VERSIONS["xgboost"]) and not args.allow_newer_xgboost:
        raise SystemExit(f"xgboost {v['xgboost']} is newer than the local {LOCAL_VERSIONS['xgboost']}: its model file "
                         f"may not load locally. Run `pip install xgboost=={LOCAL_VERSIONS['xgboost']}` first "
                         "(or pass --allow-newer-xgboost).")
    if v["torch_geometric"] != LOCAL_VERSIONS["torch_geometric"]:
        log(f"WARNING: torch_geometric {v['torch_geometric']} != local {LOCAL_VERSIONS['torch_geometric']}; "
            f"run `pip install torch_geometric=={LOCAL_VERSIONS['torch_geometric']}`")
    if args.device == "cuda" and not v["cuda_available"]:
        raise SystemExit("--device cuda but torch.cuda.is_available() is False (enable the GPU accelerator)")
    smoke = args.epochs is not None
    if smoke:
        log(f"SMOKE TEST: GAT epochs overridden to {args.epochs}; these artefacts must never be exported or reported")

    import torch
    import xgboost as xgb
    from src.evaluation.baselines import _test_signature
    from src.evaluation.data import TEST_STEPS, TRAIN_STEPS, load_elliptic
    from src.evaluation.tgat_boostnet import TGATBoostNetConfig, fit_tgat_boostnet, gat_embed
    from src.models.elliptic_training import evaluate

    git = git_info(repo)
    log("repo:", json.dumps(git))
    if not git["matches_expected"]:
        log(f"WARNING: repo HEAD {git['sha']} is not the expected {EXPECTED_REPO_SHA}")

    # ------------------------------------------------------------------ data
    data_dir = find_data_dir(args.data_dir)
    log("data dir:", data_dir)
    t0 = time.time()
    data = load_elliptic(data_dir, cache=False, verbose=True)
    load_s = time.time() - t0
    ref_meta_path = os.path.join(repo, "results/runs/tgat_boostnet_tuned/seed_42.json")
    ref_meta = json.load(open(ref_meta_path))
    sig = _test_signature(data)
    if sig != ref_meta["test_signature"]:
        raise SystemExit(f"test signature {sig} != {ref_meta['test_signature']} of the evaluated run; wrong data")
    tr, te = data.train_mask, data.test_mask
    log(f"loaded {data.n} nodes, {data.edge_index.shape[1]} edges in {load_s:.0f}s; test signature {sig} ok; "
        f"train rows {int(tr.sum())}, test rows {int(te.sum())}")

    # ------------------------------------------------------------------ config (as tuning.final_run)
    sel = json.load(open(os.path.join(repo, "results/tuning/selected.json")))["tgat_boostnet"]
    params = dict(sel["params"])
    w = params.pop("w_illicit")
    cfg = TGATBoostNetConfig(**sel["gat"], w_illicit=w, gbdt=params)
    if smoke:
        cfg = dataclasses.replace(cfg, epochs=args.epochs)
    threshold = float(sel["threshold"])
    log("config:", json.dumps(cfg.to_dict()), "threshold", threshold)

    # ------------------------------------------------------------------ train
    t0 = time.time()
    r = fit_tgat_boostnet(data, cfg, seed=args.seed, device=args.device, log=log)
    fit_s = time.time() - t0
    gat, booster, X_tab, X_gat, E = r["gat_model"], r["booster"], r["X_tab"], r["X_gat"], r["edges"]
    p_test_gpu = np.asarray(r["p_test"], dtype=np.float32)
    emb_test_gpu = np.asarray(r["emb_test"], dtype=np.float32)
    log(f"fit_tgat_boostnet done in {fit_s:.0f}s")

    # standardisation statistics: the _standardise_on formula on the labelled training rows it uses
    fit_rows = data.mask(TRAIN_STEPS)
    mu = data.X_raw[fit_rows].astype(np.float64).mean(0)
    sd = data.X_raw[fit_rows].astype(np.float64).std(0)
    sd[sd == 0] = 1.0
    if not np.array_equal(((data.X_raw - mu) / sd).astype(np.float32), X_tab[:, :165]):
        raise SystemExit("recomputed standardisation is not bit-equal to fit_tgat_boostnet's X_tab")
    x_test_sha = sha256_array(X_tab[te])

    # ------------------------------------------------------------------ save
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; refusing to overwrite (one-run rule)")
    os.makedirs(out)
    files = {}

    def path(name):
        files[name] = os.path.join(out, name)
        return files[name]

    sd_cpu = {k: t.detach().cpu().clone() for k, t in gat.state_dict().items()}
    torch.save({"config": dict(gat.config), "state_dict": sd_cpu}, path("gat.pt"))
    np.savez(path("gat_weights.npz"), **{k: t.numpy() for k, t in sd_cpu.items()})
    booster.save_model(path("tgat_boostnet_xgb.ubj"))
    booster.save_model(path("tgat_boostnet_xgb.json"))
    np.savez(path("scaler_stats.npz"), mu=mu, sd=sd, x_test_std_sha256=np.array(x_test_sha))

    # CPU replay on the same machine: GAT deep-copied to CPU, booster reloaded from .ubj into a CPU classifier
    t0 = time.time()
    gat_cpu = copy.deepcopy(gat).to("cpu").eval()
    test_steps = (data.time_steps >= TEST_STEPS[0]) & (data.time_steps <= TEST_STEPS[1])
    idx, _, emb = gat_embed(gat_cpu, X_gat, E, test_steps, "cpu")
    emb_all = np.zeros((data.n, cfg.emb_dim), dtype=np.float32)
    emb_all[idx] = emb
    emb_test_cpu = emb_all[te]
    cpu_booster = xgb.XGBClassifier()
    cpu_booster.load_model(files["tgat_boostnet_xgb.ubj"])
    cpu_booster.set_params(device="cpu")
    cpu_booster.get_booster().set_param({"device": "cpu"})
    p_test_cpu = cpu_booster.predict_proba(np.hstack([X_tab[te], emb_test_cpu]))[:, 1].astype(np.float32)
    p_xgbcpu_embgpu = cpu_booster.predict_proba(np.hstack([X_tab[te], emb_test_gpu]))[:, 1].astype(np.float32)
    cpu_s = time.time() - t0
    np.savez(path("test_outputs.npz"), tx_ids=data.tx_ids[te], time_step=data.time_steps[te], y=data.y[te],
             p_test_gpu=p_test_gpu, emb_test_gpu=emb_test_gpu, emb_test_cpu=emb_test_cpu, p_test_cpu=p_test_cpu,
             p_xgbcpu_embgpu=p_xgbcpu_embgpu)

    # ------------------------------------------------------------------ metrics + comparison
    y_te = data.y[te]
    metrics = {"threshold": threshold, "p_test_gpu": evaluate(y_te, p_test_gpu, threshold),
               "p_test_cpu": evaluate(y_te, p_test_cpu, threshold),
               "p_xgbcpu_embgpu": evaluate(y_te, p_xgbcpu_embgpu, threshold)}
    for k in ("p_test_gpu", "p_test_cpu"):
        log(f"{k}: F1 {metrics[k]['f1_illicit']:.4f}  PR-AUC {metrics[k]['average_precision']:.4f}"
            + ("  (SMOKE TEST - not a result)" if smoke else ""))
    comparison = None
    ref_npz = os.path.join(repo, "results/runs/tgat_boostnet_tuned/seed_42.npz")
    if os.path.exists(ref_npz):
        p_ref = np.load(ref_npz)["p_test"].astype(np.float32)
        m_ref = evaluate(y_te, p_ref, threshold)
        comparison = {"reference": "results/runs/tgat_boostnet_tuned/seed_42.npz", "reference_device": ref_meta.get("device"),
                      "reference_metrics": m_ref}
        for k, p in (("p_test_gpu", p_test_gpu), ("p_test_cpu", p_test_cpu)):
            comparison[k] = {"f1_diff": metrics[k]["f1_illicit"] - m_ref["f1_illicit"],
                             "pr_auc_diff": metrics[k]["average_precision"] - m_ref["average_precision"],
                             "max_abs_prob_diff": float(np.max(np.abs(p - p_ref))),
                             "decision_agreement": float(((p >= threshold) == (p_ref >= threshold)).mean())}
        log("vs evaluated Colab run:", json.dumps(comparison["p_test_gpu"]))

    meta = {
        "smoke_test": smoke,
        "smoke_test_note": "GAT epochs overridden for a plumbing test; NOT a model result, never export" if smoke else None,
        "seed": args.seed, "device": args.device, "versions": v, "local_versions_expected": LOCAL_VERSIONS,
        "repo": git, "data_dir": data_dir, "test_signature": sig,
        "train_steps": list(TRAIN_STEPS), "test_steps": list(TEST_STEPS),
        "n_train_rows": int(tr.sum()), "n_test_rows": int(te.sum()),
        "config": cfg.to_dict(), "selected": sel, "threshold": threshold, "gat_config": dict(gat.config),
        "timings_seconds": {"load_data": load_s, "fit_tgat_boostnet": fit_s, "train_seconds": r["train_seconds"],
                            "gat_full_seconds": r.get("gat_full_seconds"), "crossfit_seconds": r.get("crossfit_seconds"),
                            "gbdt_seconds": r.get("gbdt_seconds"), "cpu_replay": cpu_s,
                            "script_total": time.time() - t_script},
        "gat_loss_history": r.get("gat_loss_history"),
        "data_sha256": {"X_raw": sha256_array(data.X_raw), "tx_ids": sha256_array(data.tx_ids),
                        "y": sha256_array(data.y), "edge_index": sha256_array(data.edge_index),
                        "time_steps": sha256_array(data.time_steps)},
        "data_dtypes": {"X_raw": str(data.X_raw.dtype), "tx_ids": str(data.tx_ids.dtype), "y": str(data.y.dtype),
                        "edge_index": str(data.edge_index.dtype), "time_steps": str(data.time_steps.dtype)},
        "x_test_std_sha256": x_test_sha,
        "metrics": metrics, "comparison_with_evaluated_run": comparison,
        "files_sha256": {k: sha256_file(p) for k, p in files.items()},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    zip_path = shutil.make_archive(out, "zip", root_dir=out)
    log(f"wrote {out} and {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB) in {time.time() - t_script:.0f}s")
    log("download", zip_path)


if __name__ == "__main__":
    main()
