"""
Build the "gold" tables the Superset dashboards read (data/bi/gold/<table>/, Parquet).

Layers (docs/BI_SUPERSET.md):
  * Spark does the heavy aggregation: the 16.9 M scored Zenodo transactions (joined to their features)
    and the 203,769-node Elliptic graph are reduced to small tables here.
  * Small result tables that already exist (tables/*.csv, results/runs/*/seed_42.npz, the served
    bundle's validation files) are normalised with pandas into long, chart-friendly Parquet.
  * Live streaming outputs are NOT copied: DuckDB views read data/streaming/run_*/ directly
    (scripts/bi/build_duckdb.py), so a running pipeline shows up without rebuilding anything.

Every number comes from an existing output file; nothing is computed on the test labels that is not
already reported elsewhere (curves are recomputed from the saved test probabilities).

    .venv\\Scripts\\python.exe scripts\\build_bi_gold.py               # everything
    .venv\\Scripts\\python.exe scripts\\build_bi_gold.py --skip-spark  # only the small pandas tables
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
GOLD = os.path.join(ROOT, "data", "bi", "gold")
TABLES = os.path.join(ROOT, "tables")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_pd(df: pd.DataFrame, name: str):
    out = os.path.join(GOLD, name)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    df = df.copy()
    cols, seen = [], {}
    for c in df.columns:
        raw = str(c).replace("%", " pct").replace("±", " sd")
        k = re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_").lower() or "col"
        if k in seen:
            seen[k] += 1
            k = f"{k}_{seen[k]}"
        else:
            seen[k] = 0
        cols.append(k)
    df.columns = cols
    df.to_parquet(os.path.join(out, "part-0.parquet"), index=False)
    log(f"  {name}: {len(df)} rows")


def write_spark(sdf, name: str):
    from src.spark.session import to_spark_path
    out = os.path.join(GOLD, name)
    shutil.rmtree(out, ignore_errors=True)
    sdf.coalesce(1).write.mode("overwrite").parquet(to_spark_path(out))
    for f in glob.glob(os.path.join(out, ".*.crc")) + glob.glob(os.path.join(out, "_SUCCESS")):
        os.remove(f)
    log(f"  {name}: {sdf.sparkSession.read.parquet(to_spark_path(out)).count()} rows (Spark)")


# ----------------------------------------------------------------------------------------- Spark
def spark_tables(driver_memory: str):
    from pyspark.sql import functions as F
    from src.spark.session import get_spark, to_spark_path

    spark = get_spark("BSAD-BI-gold", master="local[2]", shuffle_partitions=8, driver_memory=driver_memory)
    try:
        # ---------------- Elliptic graph
        nodes = spark.read.parquet(to_spark_path(os.path.join(ROOT, "data/processed/elliptic_spark/nodes")))
        edges = spark.read.parquet(to_spark_path(os.path.join(ROOT, "data/processed/elliptic_spark/edges")))
        cls = F.when(F.col("label") == 1, "illicit").when(F.col("label") == 0, "licit").otherwise("unknown")
        nodes = nodes.withColumn("class", cls).withColumn("time_step", F.col("time_step").cast("int"))

        write_spark(nodes.groupBy("time_step", "class", "split").count().withColumnRenamed("count", "n"),
                    "ell_class_by_step")
        write_spark(nodes.select("tx_id", F.col("label").cast("int").alias("label"), "class", "split",
                                 "time_step"), "ell_labels")

        deg = (edges.select(F.col("source_tx_id").alias("tx_id")).union(edges.select(F.col("target_tx_id").alias("tx_id")))
               .groupBy("tx_id").agg(F.count(F.lit(1)).alias("degree")))
        nd = nodes.join(deg, "tx_id", "left").fillna({"degree": 0})
        g = nd.groupBy("time_step").agg(F.count(F.lit(1)).alias("n_nodes"), F.avg("degree").alias("mean_degree"),
                                        F.max("degree").alias("max_degree"),
                                        F.sum(F.when(F.col("class") == "illicit", 1).otherwise(0)).alias("n_illicit"),
                                        F.sum(F.when(F.col("class") == "licit", 1).otherwise(0)).alias("n_licit"))
        e = edges.groupBy(F.col("time_step").cast("int").alias("time_step")).count().withColumnRenamed("count", "n_edges")
        write_spark(g.join(e, "time_step").withColumn("illicit_share", F.col("n_illicit") / (F.col("n_illicit") + F.col("n_licit"))),
                    "ell_graph_by_step")
        # zero-padded labels so alphabetical order is numeric order
        bucket = (F.when(F.col("degree") <= 1, "00-01").when(F.col("degree") <= 2, "02").when(F.col("degree") <= 3, "03")
                  .when(F.col("degree") <= 5, "04-05").when(F.col("degree") <= 10, "06-10").when(F.col("degree") <= 50, "11-50")
                  .otherwise("51+"))
        write_spark(nd.withColumn("degree_bucket", bucket).groupBy("degree_bucket", "class").count()
                    .withColumnRenamed("count", "n"), "ell_degree_dist")
        feats = [f"f{i}" for i in range(2, 10)]
        write_spark(nodes.filter(F.col("class") != "unknown").groupBy("class")
                    .agg(*[F.avg(c).alias(c) for c in feats]), "ell_feature_profile")
        write_spark(nd.filter(F.col("class") != "unknown")
                    .sample(fraction=0.15, seed=7).select("tx_id", "class", "time_step", "degree", "f2", "f3", "f4", "f5"),
                    "ell_feature_sample")

        # ---------------- Zenodo: 16.9 M scored transactions, joined to their features
        sc = spark.read.parquet(to_spark_path(os.path.join(ROOT, "data/processed/zenodo_anomaly/scores")))
        ft = spark.read.parquet(to_spark_path(os.path.join(ROOT, "data/processed/zenodo_anomaly/features")))
        write_spark(sc.groupBy("dq_status", "score_status").count().withColumnRenamed("count", "n"), "zen_status")
        s = sc.filter(F.col("score_status") == "scored")
        lo, hi = s.agg(F.min("score"), F.max("score")).first()
        nb = 60
        width = (hi - lo) / nb
        b = F.least(F.floor((F.col("score") - lo) / width), F.lit(nb - 1)).cast("int")
        write_spark(s.withColumn("bin", b).groupBy("bin").agg(F.count(F.lit(1)).alias("n"),
                                                              F.sum(F.col("flagged").cast("int")).alias("n_flagged"))
                    .withColumn("bin_lo", F.lit(lo) + F.col("bin") * F.lit(width))
                    .withColumn("bin_hi", F.col("bin_lo") + F.lit(width))
                    .withColumn("bin_mid", (F.col("bin_lo") + F.col("bin_hi")) / 2), "zen_score_hist")
        q = F.expr("percentile_approx(score, array(0.01, 0.25, 0.5, 0.75, 0.99, 0.999), 2000)")
        write_spark(s.groupBy("block_chunk").agg(F.count(F.lit(1)).alias("n"), F.sum(F.col("flagged").cast("int")).alias("n_flagged"),
                                                 F.avg("score").alias("mean_score"), q.alias("q"))
                    .select("block_chunk", "n", "n_flagged", "mean_score",
                            (F.col("n_flagged") / F.col("n") * 1000).alias("flag_rate_per_mille"),
                            *[F.col("q")[i].alias(n) for i, n in enumerate(["p01", "p25", "p50", "p75", "p99", "p999"])]),
                    "zen_by_chunk")
        write_spark(s.withColumn("block_bucket", (F.floor(F.col("block_number") / 100) * 100).cast("long"))
                    .groupBy("block_bucket").agg(F.count(F.lit(1)).alias("n"), F.sum(F.col("flagged").cast("int")).alias("n_flagged"),
                                                 F.avg("score").alias("mean_score"),
                                                 F.expr("percentile_approx(score, 0.99, 1000)").alias("p99"))
                    .withColumn("flag_rate_per_mille", F.col("n_flagged") / F.col("n") * 1000), "zen_by_block_bucket")
        rc = (s.filter(F.col("flagged")).withColumn("reason", F.explode(F.split(F.col("reason_codes"), r"[;,|]")))
              .withColumn("reason", F.trim("reason")).filter(F.col("reason") != "")
              # "new_addr_fraction:high(+4.7)" -> feature, direction, robust z
              .withColumn("feature", F.regexp_extract("reason", r"^([^:]+):", 1))
              .withColumn("direction", F.regexp_extract("reason", r":(\w+)\(", 1))
              .withColumn("z", F.regexp_extract("reason", r"\(([-+0-9.]+)\)", 1).cast("double")))
        write_spark(rc.groupBy("feature", "direction").agg(F.count(F.lit(1)).alias("n"), F.avg("score").alias("mean_score"),
                                                           F.avg(F.abs("z")).alias("mean_abs_z")), "zen_reasons")

        j = s.select("block_chunk", "block_number", "tx_index", "score", "percentile", "flagged").join(
            ft.select("block_chunk", "block_number", "tx_index", "num_inputs", "num_outputs", "fee_rate",
                      "new_address_fraction", "avg_input_prior_tx_count", "total_output_satoshi"),
            ["block_chunk", "block_number", "tx_index"])
        dec = F.least(F.floor(F.col("percentile") * 10), F.lit(9)).cast("int") + 1
        write_spark(j.withColumn("score_decile", dec).groupBy("score_decile").agg(
            F.count(F.lit(1)).alias("n"), F.avg("num_inputs").alias("mean_inputs"), F.avg("num_outputs").alias("mean_outputs"),
            F.expr("percentile_approx(fee_rate, 0.5, 1000)").alias("median_fee_rate"),
            F.avg("new_address_fraction").alias("mean_new_address_fraction"),
            F.avg("avg_input_prior_tx_count").alias("mean_input_prior_tx"),
            F.sum(F.col("flagged").cast("int")).alias("n_flagged")), "zen_decile_profile")
        write_spark(j.filter(F.col("flagged")).union(j.filter(~F.col("flagged")).sample(fraction=0.003, seed=11))
                    .withColumn("log_inputs", F.log1p("num_inputs")).withColumn("log_outputs", F.log1p("num_outputs"))
                    .withColumn("log_output_btc", F.log10(F.col("total_output_satoshi") / 1e8 + 1e-8)),
                    "zen_sample")
    finally:
        spark.stop()


# ----------------------------------------------------------------------------------------- pandas
FAMILY = {"TGAT": "Ours (GAT + GBDT)", "GAT-BoostNet": "Ours (GAT + GBDT)"}


def family(model: str) -> str:
    m = model.lower()
    if "tgat" in m or "ours" in m:
        return "Ours: GNN + GBDT"
    if any(k in m for k in ("gcn + xgb", "gat + rf", "gcn emb", "gat emb")):
        return "GNN embedding + trees"
    if any(k in m for k in ("gcn", "gat", "graphsage", "evolvegcn")):
        return "End-to-end GNN"
    if any(k in m for k in ("xgb", "lightgbm", "catboost", "gbdt", "random forest", "rf")):
        return "Tree ensemble"
    if any(k in m for k in ("mlp",)):
        return "Neural (tabular)"
    return "Linear / kernel"


def offline_tables():
    mean = pd.read_csv(os.path.join(TABLES, "03_leaderboard_all_metrics_mean.csv")).rename(columns={"Unnamed: 0": "model"})
    std = pd.read_csv(os.path.join(TABLES, "03_leaderboard_all_metrics_std.csv")).rename(columns={"Unnamed: 0": "model"})
    lm = mean.melt(id_vars="model", var_name="metric", value_name="mean")
    ls = std.melt(id_vars="model", var_name="metric", value_name="std")
    lb = lm.merge(ls, on=["model", "metric"], how="left")
    lb["family"] = lb.model.map(family)
    lb["is_ours"] = lb.model.str.contains("TGAT", case=False)
    write_pd(lb, "model_metrics_long")
    wide = mean.copy()
    wide["family"] = wide.model.map(family)
    wide["is_ours"] = wide.model.str.contains("TGAT", case=False)
    write_pd(wide, "model_metrics_wide")

    tu = pd.read_csv(os.path.join(TABLES, "04_tuned_vs_untuned.csv")).rename(columns={"Unnamed: 0": "model"})
    rows = []
    for _, r in tu.iterrows():
        name = r["model"]
        kind = "tuned" if "(tuned)" in name else "untuned"
        fam = name.replace(" (tuned)", "").replace(" (untuned)", "")
        rec = {"model_family": fam, "kind": kind, "threshold": r["threshold"]}
        for c in tu.columns[2:]:
            m = re.match(r"\s*([0-9.]+)\s*±\s*([0-9.]+)", str(r[c]))
            key = re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower()
            rec[key] = float(m.group(1)) if m else np.nan
            rec[key + "_sd"] = float(m.group(2)) if m else np.nan
        rows.append(rec)
    write_pd(pd.DataFrame(rows), "tuned_vs_untuned")

    ps = pd.read_csv(os.path.join(TABLES, "03_per_timestep_f1.csv"))
    pl = ps.melt(id_vars="time_step", var_name="model", value_name="f1")
    pl["family"] = pl.model.map(family)
    pl["is_ours"] = pl.model.str.contains("TGAT", case=False)
    pl["period"] = np.where(pl.time_step < 43, "steps 35-42 (before shift)", "steps 43-49 (after shift)")
    write_pd(pl, "per_step_f1_long")

    eff = pd.read_csv(os.path.join(TABLES, "03_efficiency.csv")).rename(columns={"Unnamed: 0": "model"})
    eff["family"] = eff.model.map(family)
    eff["is_ours"] = eff.model.str.contains("TGAT", case=False)
    write_pd(eff, "model_efficiency")

    for name in ["03_significance", "03_ablation", "03_reproduced_vs_reported", "03_sensitivity", "03_rank_summary",
                 "01_spark_scaling", "01_class_balance", "01_preprocessing_metrics", "01_quality_before_after",
                 "02_shap_top30", "02_learning_curve", "02_before_after_step43", "04_significance_tuned",
                 "01_before_after_preprocessing", "02_course_metrics", "03_reported_in_paper", "01A_zenodo_scaling"]:
        p = os.path.join(TABLES, name + ".csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = df.rename(columns={"Unnamed: 0": "item"})
            write_pd(df, "t" + name)

    # significance: parse the delta-F1 CI into numbers for a CI chart
    # 03_significance compares the a-priori (untuned) TGAT-BoostNet with each baseline (notebook 03)
    sig = pd.read_csv(os.path.join(TABLES, "03_significance.csv")).rename(columns={"Unnamed: 0": "baseline"})
    cols = {c: re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in sig.columns}
    sig = sig.rename(columns=cols)
    ci_col = next((c for c in sig.columns if "ci" in c and "f1" in c), None)
    if ci_col:
        lohi = sig[ci_col].astype(str).str.extract(r"\[?\s*([-+]?[0-9.]+)\s*,\s*([-+]?[0-9.]+)")
        sig["ci_lo"] = pd.to_numeric(lohi[0], errors="coerce")
        sig["ci_hi"] = pd.to_numeric(lohi[1], errors="coerce")
    write_pd(sig, "significance_ci")


def curve_tables():
    from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
    from src.evaluation.data import load_elliptic
    data = load_elliptic(verbose=False)
    y = data.y[data.test_mask]
    names = {"tgat_boostnet_tuned": "TGAT-BoostNet (tuned, ours)", "lgbm_tuned": "LightGBM (tuned)",
             "rf_tuned": "Random Forest (tuned)", "xgb_tuned": "XGBoost (tuned)", "catboost": "CatBoost",
             "gcn_xgb": "GCN emb. + XGBoost", "gat_rf": "GAT emb. + RF", "gcn": "GCN", "gat": "GAT alone",
             "graphsage": "GraphSAGE", "mlp": "MLP", "lr": "Logistic Regression", "evolvegcn": "EvolveGCN-O"}
    pr, roc, cal, summ = [], [], [], []
    for key, label in names.items():
        f = os.path.join(ROOT, "results/runs", key, "seed_42.npz")
        if not os.path.exists(f):
            f = os.path.join(ROOT, "results/runs", key, "seed_0.npz")
            if not os.path.exists(f):
                continue
        p = np.load(f)["p_test"]
        if len(p) != len(y):
            continue
        # common grids so every model has a value at every x (continuous lines in the dashboard)
        grid = np.round(np.linspace(0, 1, 101), 2)
        prec, rec, _ = precision_recall_curve(y, p)
        # interpolated precision: best precision achievable at recall >= r (standard PR interpolation)
        pr += [{"model": label, "recall": float(r), "precision": float(prec[rec >= r].max())} for r in grid]
        fpr, tpr, _ = roc_curve(y, p)
        roc += [{"model": label, "fpr": float(f), "tpr": float(np.interp(f, fpr, tpr))} for f in grid]
        summ.append({"model": label, "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
                     "is_ours": "ours" in label})
        if key in ("tgat_boostnet_tuned", "lgbm_tuned", "rf_tuned"):
            bins = np.clip((p * 10).astype(int), 0, 9)
            for b in range(10):
                m = bins == b
                if m.sum():
                    cal.append({"model": label, "bin": b, "bin_centre": (b + 0.5) / 10, "mean_predicted": float(p[m].mean()),
                                "observed_illicit_rate": float(y[m].mean()), "n": int(m.sum())})
    write_pd(pd.DataFrame(pr), "curve_pr")
    write_pd(pd.DataFrame(roc), "curve_roc")
    write_pd(pd.DataFrame(cal), "curve_calibration")
    write_pd(pd.DataFrame(summ), "curve_summary")


def served_tables():
    v = os.path.join(ROOT, "data/validation/tgat_boostnet_bundle")
    bc = json.load(open(os.path.join(v, "bundle_checks.json")))
    m = bc["test_metrics_from_bundle"]
    cm = [{"actual": "illicit", "predicted": "illicit", "cell": "TP", "n": m["tp"]},
          {"actual": "illicit", "predicted": "licit", "cell": "FN", "n": m["fn"]},
          {"actual": "licit", "predicted": "illicit", "cell": "FP", "n": m["fp"]},
          {"actual": "licit", "predicted": "licit", "cell": "TN", "n": m["tn"]}]
    write_pd(pd.DataFrame(cm), "served_confusion")
    old = json.load(open(os.path.join(ROOT, "artifacts/models/elliptic/3.0.0/manifest.json")))["test_metrics_seed123"]
    rows = []
    for name, mm in [("4.0.0 TGAT-BoostNet (served)", m), ("3.0.0 GAT-BoostNet (previous)", old)]:
        for k, lab in [("precision", "Precision"), ("recall", "Recall"), ("f1_illicit", "F1"), ("roc_auc", "ROC-AUC"),
                       ("average_precision", "PR-AUC")]:
            rows.append({"bundle": name, "metric": lab, "value": mm[k]})
    write_pd(pd.DataFrame(rows), "served_vs_previous")
    rep = json.load(open(os.path.join(v, "reproduction.json")))
    write_pd(pd.DataFrame([{"check": k, "passed": bool(c.get("pass"))} for k, c in rep.items()
                           if isinstance(c, dict) and "pass" in c]), "served_checks")
    ts = pd.read_parquet(os.path.join(v, "test_scores.parquet"))
    sc = "score_local_cpu"
    thr = bc["threshold"]
    ts["class"] = np.where(ts.label == 1, "illicit", "licit")
    ts["flagged"] = ts[sc] >= thr
    edges = np.linspace(0, 1, 41)
    ts["bin"] = np.clip(np.digitize(ts[sc], edges) - 1, 0, 39)
    h = ts.groupby(["bin", "class"]).size().rename("n").reset_index()
    h["bin_mid"] = (h["bin"] + 0.5) / 40
    h["signed_n"] = np.where(h["class"] == "illicit", h.n, -h.n)
    write_pd(h, "served_score_hist")
    rows = []
    for st, g in ts.groupby("time_step"):
        yy, ff = g.label == 1, g.flagged
        tp, fp, fn = int((yy & ff).sum()), int((~yy & ff).sum()), int((yy & ~ff).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"time_step": int(st), "illicit": int(yy.sum()), "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0})
    write_pd(pd.DataFrame(rows), "served_per_step")
    thr_rows = []
    for t in np.round(np.linspace(0.05, 0.95, 19), 2):
        ff = ts[sc] >= t
        yy = ts.label == 1
        tp, fp, fn = int((yy & ff).sum()), int((~yy & ff).sum()), int((yy & ~ff).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        for k, val in [("precision", p), ("recall", r), ("f1", 2 * p * r / (p + r) if p + r else 0.0)]:
            thr_rows.append({"threshold": float(t), "metric": k, "value": val})
    write_pd(pd.DataFrame(thr_rows), "served_threshold_sweep")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-spark", action="store_true")
    ap.add_argument("--driver-memory", default="2g")
    args = ap.parse_args()
    os.makedirs(GOLD, exist_ok=True)
    t0 = time.time()
    log("offline result tables")
    offline_tables()
    log("curves from saved test probabilities")
    curve_tables()
    log("served bundle tables")
    served_tables()
    if not args.skip_spark:
        log("Spark aggregations (Elliptic graph, 16.9 M Zenodo transactions)")
        spark_tables(args.driver_memory)
    log(f"done in {time.time() - t0:.0f}s -> {GOLD}")


if __name__ == "__main__":
    main()
