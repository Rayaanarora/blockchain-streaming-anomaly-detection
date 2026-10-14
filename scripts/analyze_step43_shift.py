"""
Post-hoc diagnostic of the Elliptic test-period distribution shift around time step 43.

This script is a *diagnostic*, run after model selection. It reads test labels only to describe
the test population (prevalence, which population moved); nothing here feeds back into training,
hyper-parameters, thresholds or model choice.

Measurements
  1. Labelled counts / illicit prevalence per test step, and pre (35-42) vs post (43-49) totals.
  2. Does the illicit population after step 43 still look like training-period illicit activity?
     For each labelled test transaction, the fraction of illicit labels among its k=10 nearest
     training transactions (steps 1-29, standardised local features f1-f93). Compared for illicit
     and licit transactions in 35-42 vs 43-49. Plus per-class standardised mean differences (SMD)
     against the training class means. (A domain classifier was tried first and separated *every*
     pair of periods with AUC ~1.0, including licit-vs-licit, so it does not isolate step 43.)
  3. Graph statistics per step (mean degree of illicit / licit nodes; edges are intra-step).
  4. Per-step test recall/precision of the selected model (mean over seeds), if the final
     selection predictions exist.

The Elliptic data contains no wall-clock timestamps, so this script cannot attribute the shift to
a specific real-world event. Literature (Weber et al., 2019) links the post-43 degradation to a
dark-market shutdown; that is reported as an external hypothesis only.

Usage:
    .venv\\Scripts\\python.exe scripts\\analyze_step43_shift.py
Output: data/validation/final_model_selection/step43_shift/
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.elliptic_training import load_spark_output  # noqa: E402

PRE, POST = (35, 42), (43, 49)


def knn_illicit_fraction(X_ref, y_ref, X_q, k=10):
    nn = NearestNeighbors(n_neighbors=k).fit(X_ref)
    _, idx = nn.kneighbors(X_q)
    return y_ref[idx].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark-dir", default=os.path.join(ROOT, "data/processed/elliptic_spark"))
    ap.add_argument("--selection-dir", default=os.path.join(ROOT, "data/validation/final_model_selection"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.selection_dir, "step43_shift")
    os.makedirs(out, exist_ok=True)

    d = load_spark_output(args.spark_dir)
    t, y, X = d.time_steps, d.y, d.X_raw
    lab = y >= 0

    # 1. prevalence
    rows = []
    for s in range(35, 50):
        m = (t == s) & lab
        rows.append({"time_step": s, "labelled": int(m.sum()), "illicit": int((y[m] == 1).sum()),
                     "prevalence": float((y[m] == 1).mean()) if m.any() else np.nan})
    prev = pd.DataFrame(rows)

    def window(w):
        return (t >= w[0]) & (t <= w[1])

    def period(w):
        m = window(w) & lab
        return {"labelled": int(m.sum()), "illicit": int((y[m] == 1).sum()), "prevalence": float((y[m] == 1).mean())}

    # 2. similarity of test transactions to training-period classes
    train = (t <= 29) & lab
    mu, sd = X[train, :93].mean(0), X[train, :93].std(0) + 1e-9
    Z = (X[:, :93] - mu) / sd
    shift = []
    for cls, name in ((1, "illicit"), (0, "licit")):
        ref_mean = Z[train & (y == cls)].mean(0)
        for w, wname in ((PRE, "35-42"), (POST, "43-49")):
            q = window(w) & lab & (y == cls)
            frac = knn_illicit_fraction(Z[train], y[train], Z[q])
            smd = np.abs(Z[q].mean(0) - ref_mean)
            shift.append({"class": name, "window": wname, "n": int(q.sum()),
                          "knn10_illicit_fraction_mean": float(frac.mean()),
                          "share_with_majority_illicit_neighbours": float((frac > 0.5).mean()),
                          "median_abs_smd_vs_train_class": float(np.median(smd)),
                          "features_abs_smd_gt_0.5": int((smd > 0.5).sum())})
    shift = pd.DataFrame(shift)

    # 3. graph degree per step
    s_idx, t_idx = d.edge_index
    deg = np.bincount(s_idx, minlength=len(y)) + np.bincount(t_idx, minlength=len(y))
    gstats = []
    for s in range(1, 50):
        m = t == s
        gstats.append({"time_step": s, "mean_degree_illicit": float(deg[m & (y == 1)].mean()) if (m & (y == 1)).any() else np.nan,
                       "mean_degree_licit": float(deg[m & (y == 0)].mean()),
                       "mean_degree_all": float(deg[m].mean())})
    gstats = pd.DataFrame(gstats)

    # 4. per-step performance of the selected model
    perf = None
    sel_path = os.path.join(args.selection_dir, "selection.json")
    step_path = os.path.join(args.selection_dir, "per_step_metrics.csv")
    selected = None
    if os.path.exists(sel_path) and os.path.exists(step_path):
        selected = json.load(open(sel_path))["selected_config"]
        ps = pd.read_csv(step_path)
        ps = ps[ps.config == selected]
        perf = ps.groupby("time_step").agg(recall_mean=("recall", "mean"), recall_std=("recall", "std"),
                                           precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
                                           tp=("tp", "mean"), fn=("fn", "mean"), fp=("fp", "mean")).reset_index()
        agg = ps.groupby("seed").apply(
            lambda g: pd.Series({w: (g[(g.time_step >= lo) & (g.time_step <= hi)].tp.sum()
                                     / max(1, (g[(g.time_step >= lo) & (g.time_step <= hi)][["tp", "fn"]].sum().sum())))
                                 for w, (lo, hi) in (("recall_35_42", PRE), ("recall_43_49", POST))}),
            include_groups=False)
        perf_periods = {k: {"mean": float(agg[k].mean()), "std": float(agg[k].std())} for k in agg.columns}
    prev.to_csv(os.path.join(out, "prevalence_per_step.csv"), index=False)
    shift.to_csv(os.path.join(out, "class_similarity_shift.csv"), index=False)
    gstats.to_csv(os.path.join(out, "graph_degree_per_step.csv"), index=False)
    if perf is not None:
        perf.to_csv(os.path.join(out, "selected_model_per_step.csv"), index=False)

    report = {
        "periods": {"pre_35_42": period(PRE), "post_43_49": period(POST)},
        "class_similarity_shift": shift.to_dict("records"),
        "selected_config": selected,
        "selected_model_recall_by_period": perf_periods if perf is not None else None,
        "note": "Post-hoc diagnostic; test labels describe the population only and are not used for any model "
                "decision. No timestamps exist in Elliptic, so no real-world cause can be established from the data.",
    }
    json.dump(report, open(os.path.join(out, "step43_shift.json"), "w"), indent=2)
    print(prev.to_string(index=False))
    print(shift.round(4).to_string(index=False))
    print(json.dumps(report["periods"], indent=1))
    if perf is not None:
        print(perf.round(3).to_string(index=False))
        print(json.dumps(perf_periods, indent=1))


if __name__ == "__main__":
    main()
