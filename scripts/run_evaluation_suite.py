"""
Compute (and cache) the model runs used by notebooks 02/03, outside Jupyter.

The notebooks call the same function (src.evaluation.baselines.run) and reuse this cache, so the
heavy GNN training can be spread over parallel worker processes:

    .venv\\Scripts\\python.exe scripts\\run_evaluation_suite.py --models tgat_boostnet --threads 5
    .venv\\Scripts\\python.exe scripts\\run_evaluation_suite.py --models gcn skip_gcn graphsage --threads 4
    .venv\\Scripts\\python.exe scripts\\run_evaluation_suite.py --group ablations --threads 5
    .venv\\Scripts\\python.exe scripts\\run_evaluation_suite.py --quick --models all   # smoke test, 3 epochs

Idle-machine timing pass (run alone, nothing else on the CPU; notebooks read results/timing/ for timings):

    .venv\\Scripts\\python.exe scripts\\run_evaluation_suite.py --group comparison --seeds 42 --threads 8 --results-dir results/timing

--quick writes to results_quick/ so it never mixes with real results.
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=[])
ap.add_argument("--group", choices=["comparison", "ablations", "all"], default=None)
ap.add_argument("--seeds", type=int, nargs="*", default=None)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--device", default="auto")
ap.add_argument("--force", action="store_true")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--results-dir", default=None,
                help="cache root; the idle-machine timing pass uses results/timing (read by notebooks 02/03)")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = str(args.threads)
os.environ["BSAD_N_JOBS"] = str(args.threads)
if args.results_dir:
    os.environ["BSAD_RESULTS_DIR"] = os.path.abspath(args.results_dir)
if args.quick:
    os.environ["BSAD_RESULTS_DIR"] = os.path.join(ROOT, "results_quick")
sys.path.insert(0, ROOT)

import torch  # noqa: E402

torch.set_num_threads(args.threads)
from src.evaluation import baselines as B  # noqa: E402
from src.evaluation.data import load_elliptic  # noqa: E402
from src.evaluation.tgat_boostnet import get_device  # noqa: E402

keys = list(args.models)
if args.group in ("comparison", "all") or keys == ["all"]:
    keys = [k for k in keys if k != "all"] + B.COMPARISON
if args.group in ("ablations", "all") or args.models == ["all"]:
    keys += list(B.ABLATIONS)
keys = list(dict.fromkeys(keys))
device = get_device() if args.device == "auto" else args.device
data = load_elliptic()
print(f"models={keys} device={device} threads={args.threads} results={B.RESULTS_DIR}", flush=True)
for k in keys:
    for s in (args.seeds or B.seeds_for(k)):
        t0 = time.time()
        r = B.run(k, data, s, device, force=args.force, quick=args.quick, log=lambda *a: print(*a, flush=True))
        from src.evaluation.metrics import full_metrics
        m = full_metrics(data.y[data.test_mask], r["p_test"])
        print(f"[{time.strftime('%H:%M:%S')}] {k} seed={s} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} "
              f"AP={m['pr_auc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
