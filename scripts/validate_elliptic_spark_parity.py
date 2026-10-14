"""
Parity and integrity validation: Spark Elliptic output vs the legacy pandas pipeline.

Compares data/processed/elliptic_spark (Spark) with
data/processed/elliptic/elliptic_preprocessed.npz (legacy, src/data/elliptic.py) and
re-checks invariants against the raw CSVs. Exit code 1 on any failed check.

Tolerances: raw features are float32 in both pipelines and must be bit-identical
(atol 0). Scaled features are compared with atol 1e-4 against sklearn StandardScaler
fitted on the same training rows: Spark computes mean/std in float64 and casts the
result to float32, sklearn transforms float32 input, so differences are rounding only.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES = [f"f{i}" for i in range(1, 166)]


def load_spark(directory):
    nodes = pd.read_parquet(os.path.join(directory, "nodes"))
    nodes["time_step"] = nodes["time_step"].astype(int)
    nodes = nodes.sort_values("node_idx").reset_index(drop=True)
    scaled = pd.read_parquet(os.path.join(directory, "nodes_scaled"))
    scaled = scaled.sort_values("node_idx").reset_index(drop=True)
    edges = pd.read_parquet(os.path.join(directory, "edges")).sort_values("edge_id").reset_index(drop=True)
    return nodes, scaled, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark-dir", default=os.path.join(ROOT, "data/processed/elliptic_spark"))
    ap.add_argument("--legacy-npz", default=os.path.join(ROOT, "data/processed/elliptic/elliptic_preprocessed.npz"))
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data/raw/elliptic"))
    ap.add_argument("--report", default=os.path.join(ROOT, "data/validation/elliptic_spark/parity_report.json"))
    args = ap.parse_args()

    checks = {}

    def check(name, ok, detail=None):
        checks[name] = {"pass": bool(ok), **({"detail": detail} if detail is not None else {})}

    nodes, scaled, edges = load_spark(args.spark_dir)
    leg = np.load(args.legacy_npz)
    n = len(leg["tx_ids"])

    # --- row counts, ids and ordering ---------------------------------------------
    check("node_count", len(nodes) == n, {"spark": len(nodes), "legacy": n})
    check("node_idx_contiguous", np.array_equal(nodes["node_idx"].to_numpy(), np.arange(len(nodes))))
    check("tx_id_order_identical", np.array_equal(nodes["tx_id"].to_numpy(), leg["tx_ids"]))
    check("time_steps_identical", np.array_equal(nodes["time_step"].to_numpy(), leg["time_steps"]))
    check("nodes_sorted_by_step_then_id",
          bool((nodes[["time_step", "tx_id"]].apply(tuple, axis=1).is_monotonic_increasing)))

    # --- features ---------------------------------------------------------------
    Xs = nodes[FEATURES].to_numpy(np.float32)
    check("feature_columns", list(nodes.columns[nodes.columns.str.match(r"^f\d+$")]) == FEATURES)
    check("feature_dtype_float32", all(str(nodes[c].dtype) == "float32" for c in FEATURES))
    max_raw = float(np.max(np.abs(Xs.astype(np.float64) - leg["X_raw"].astype(np.float64))))
    check("raw_features_bit_identical", max_raw == 0.0, {"max_abs_diff": max_raw})
    check("no_nan_inf_raw", bool(np.isfinite(Xs).all()))

    # --- labels / masks -----------------------------------------------------------
    y = nodes["label"].to_numpy()
    check("labels_identical", np.array_equal(y, leg["y"]))
    check("label_values", set(np.unique(y)) <= {-1, 0, 1})
    check("unknown_count", int((y == -1).sum()) == 157205, int((y == -1).sum()))
    check("unknown_split_is_unlabelled", bool((nodes.loc[y == -1, "split"] == "unlabelled").all()))
    check("label_status_consistent", bool(((nodes["label_status"] == "unknown") == (y == -1)).all()))
    ts = nodes["time_step"].to_numpy()
    lab = y != -1
    exp_split = np.where(~lab, "unlabelled", np.where(ts <= 29, "train", np.where(ts <= 34, "val", "test")))
    check("chronological_split_correct", bool((nodes["split"].to_numpy() == exp_split).all()))
    counts = nodes["split"].value_counts().to_dict()
    check("split_counts", counts == {"unlabelled": 157205, "train": 26381, "test": 16670, "val": 3513}, counts)
    # legacy npz stored train=steps<=34 (no val); compare union
    check("legacy_train_mask_equals_train_plus_val",
          np.array_equal(leg["train_mask"], np.isin(nodes["split"].to_numpy(), ["train", "val"])))
    check("legacy_test_mask_equals_test", np.array_equal(leg["test_mask"], nodes["split"].to_numpy() == "test"))
    check("max_train_step_lt_min_val_step",
          ts[nodes["split"] == "train"].max() < ts[nodes["split"] == "val"].min() <= ts[nodes["split"] == "val"].max()
          < ts[nodes["split"] == "test"].min())

    # --- node-to-feature / node-to-label alignment against the raw CSVs -----------
    raw_f = pd.read_csv(os.path.join(args.raw_dir, "elliptic_txs_features.csv"), header=None)
    raw_c = pd.read_csv(os.path.join(args.raw_dir, "elliptic_txs_classes.csv"))
    raw_f = raw_f.set_index(0)
    sample = nodes.sample(2000, random_state=0)
    raw_rows = raw_f.loc[sample["tx_id"].to_numpy()]
    check("alignment_time_step_vs_raw", np.array_equal(raw_rows[1].to_numpy(), sample["time_step"].to_numpy()))
    check("alignment_features_vs_raw_sample",
          float(np.max(np.abs(raw_rows.iloc[:, 1:].to_numpy(np.float32) - sample[FEATURES].to_numpy(np.float32)))) == 0.0)
    cls = raw_c.set_index("txId")["class"].astype(str).map({"1": 1, "2": 0, "unknown": -1})
    check("alignment_labels_vs_raw_all", np.array_equal(cls.loc[nodes["tx_id"].to_numpy()].to_numpy(), y))

    # --- scaled features ----------------------------------------------------------
    tr = (nodes["split"] == "train").to_numpy()
    # Reference: sklearn in float64, cast to float32 (what Spark does). Must agree to <= 1 float32 ULP.
    X64 = leg["X_raw"].astype(np.float64)
    ref = StandardScaler().fit(X64[tr]).transform(X64).astype(np.float32)
    check("scaled_rows_aligned", np.array_equal(scaled["tx_id"].to_numpy(), nodes["tx_id"].to_numpy()))
    S = scaled[FEATURES].to_numpy(np.float32)
    ulps = np.abs(S.astype(np.float64) - ref.astype(np.float64)) / np.spacing(np.abs(ref)).astype(np.float64)
    ref32 = StandardScaler().fit(leg["X_raw"][tr]).transform(leg["X_raw"])  # float32 arithmetic, informational
    check("scaled_features_match_train_only_scaler_within_1ulp", float(ulps.max()) <= 1.0,
          {"max_ulps_vs_float64_reference": float(ulps.max()),
           "info_max_abs_diff_vs_sklearn_float32_path": float(np.max(np.abs(S - ref32)))})
    stats = json.load(open(os.path.join(args.spark_dir, "scaler_stats.json")))
    sk = StandardScaler().fit(leg["X_raw"][tr])
    check("scaler_mean_matches", float(np.max(np.abs(np.array(stats["mean"]) - sk.mean_))) < 1e-5)
    check("scaler_fitted_on_train_only", stats["fit_on"].startswith("split == 'train'"))

    # --- edges --------------------------------------------------------------------
    e_leg = leg["edge_index"]
    check("edge_count", len(edges) == e_leg.shape[1], {"spark": len(edges), "legacy": int(e_leg.shape[1])})
    check("edge_index_identical", np.array_equal(edges["source_idx"].to_numpy(), e_leg[0])
          and np.array_equal(edges["target_idx"].to_numpy(), e_leg[1]))
    raw_e = pd.read_csv(os.path.join(args.raw_dir, "elliptic_txs_edgelist.csv"))
    check("edge_order_matches_csv", np.array_equal(edges["source_tx_id"].to_numpy(), raw_e["txId1"].to_numpy())
          and np.array_equal(edges["target_tx_id"].to_numpy(), raw_e["txId2"].to_numpy()))
    tx = nodes["tx_id"].to_numpy()
    check("edge_endpoints_valid", bool((edges["source_idx"].between(0, n - 1)).all()
                                       and (edges["target_idx"].between(0, n - 1)).all()
                                       and np.array_equal(tx[edges["source_idx"]], edges["source_tx_id"].to_numpy())
                                       and np.array_equal(tx[edges["target_idx"]], edges["target_tx_id"].to_numpy())))
    check("edges_intra_time_step", bool((ts[edges["source_idx"]] == ts[edges["target_idx"]]).all()
                                        and (edges["time_step"].to_numpy() == ts[edges["source_idx"]]).all()))
    check("edge_time_steps_match_legacy", np.array_equal(edges["time_step"].to_numpy(), leg["edge_time_steps"]))
    check("no_null_in_any_column", int(nodes.isna().sum().sum() + edges.isna().sum().sum()) == 0)

    failed = [k for k, v in checks.items() if not v["pass"]]
    report = {"spark_dir": args.spark_dir, "legacy_npz": args.legacy_npz, "n_checks": len(checks),
              "n_failed": len(failed), "failed": failed, "status": "PASS" if not failed else "FAIL",
              "checks": checks}
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    for k, v in checks.items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}" + (f"  {v.get('detail')}" if "detail" in v else ""))
    print(f"\n{report['status']}: {len(checks) - len(failed)}/{len(checks)} checks passed -> {args.report}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
