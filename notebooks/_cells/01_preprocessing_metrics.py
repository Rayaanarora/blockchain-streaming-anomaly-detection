# %% [markdown]
# # 01 · Preprocessing performance metrics
# **Course 21CSC314P (SRMIST) · Distributed Streaming Anomaly Detection on Blockchain Transaction Graphs**
#
# **Main part — the Elliptic Spark pipeline that feeds TGAT-BoostNet** (`src/spark/elliptic_spark.py`): parse and
# validate the three CSVs, join labels, order nodes, map edges, chronological split, train-only standardisation,
# Parquet output. It is checked against an **independent pandas/NumPy reference** (`src/evaluation/data.py`,
# float64 standardisation).
#
# **Appendix — the Zenodo Spark pipeline** (16.9 M raw Bitcoin transactions) against the repository's pure-Python
# causal reference, on a raw slice, plus the 50k / 100k / full scaling reports. Zenodo feeds only the unsupervised
# anomaly model, so it is kept separate from the TGAT-BoostNet results.
#
# | Section | What is measured |
# |---|---|
# | 1–2 | per-feature RMSE / MAE / max error / relative error / R² / exact-match rate / CCC vs the reference |
# | 3 | null rate, malformed lines, dangling / duplicate edges, dropped rows |
# | 4 | data quality before vs after: missingness, outliers, skewness / kurtosis, class SNR, transforms, class imbalance |
# | 5 | leakage: train-only scaler, future-perturbation test, graph causality |
# | 6 | efficiency: runtime, throughput, peak memory, disk, scaling over step subsets |
# | 7 | downstream effect: classifiers on raw vs preprocessed features |
# | A | Zenodo appendix |
#
# **Runtime:** ~5 min with cached benchmarks; ~15 min when the Spark jobs are re-run (`RUN_SPARK = True`, needs Java).

# %% INCLUDE_SETUP

# %%
RUN_SPARK = IN_COLAB or os.environ.get("BSAD_RUN_SPARK") == "1"   # re-run the Spark jobs (else use cached outputs)
if RUN_SPARK and IN_COLAB:
    import shutil as _sh
    if _sh.which("java") is None:
        subprocess.run("apt-get -qq install -y openjdk-17-jre-headless > /dev/null", shell=True, check=True)
    os.environ["BSAD_JAVA_HOME"] = os.path.dirname(os.path.dirname(os.path.realpath(_sh.which("java"))))
    subprocess.run([sys.executable, "scripts/fetch_spark_jars.py"], check=True)
from src.evaluation import preprocessing as PP
from src.evaluation.data import load_elliptic, find_or_download_elliptic, standardise, cross_step_edges, SEEDS
PRE_DIR = os.path.join(os.environ.get("BSAD_RESULTS_DIR", os.path.join(ROOT, "results")), "preprocessing")
os.makedirs(PRE_DIR, exist_ok=True)
def cached_json(name, fn, rerun):
    path = os.path.join(PRE_DIR, name + ".json")
    if os.path.exists(path) and not rerun:
        return json.load(open(path))
    out = fn()
    json.dump(out, open(path, "w"), indent=1, default=float)
    return out

# %% [markdown]
# ## 1. Reference and Spark outputs

# %%
RAW = find_or_download_elliptic()
ref = load_elliptic(RAW)
SPARK_DIR = os.path.join(ROOT, "data", "processed", "elliptic_spark")
if RUN_SPARK or not os.path.exists(os.path.join(SPARK_DIR, "nodes")):
    from src.spark.session import get_spark
    from src.spark.elliptic_spark import run as spark_run
    spark = get_spark("EllipticNotebook01")
    try:
        spark_run(spark, RAW, SPARK_DIR, overwrite=True)
    finally:
        spark.stop()
S = PP.read_spark_elliptic(SPARK_DIR)
nodes, scaled = S["nodes"], S["scaled"]
FEATS = [f"f{i}" for i in range(1, 166)]
# Align Spark rows to the reference order by tx_id (both sort by time step then tx id; we verify instead of assuming)
order_identical = bool((nodes.tx_id.to_numpy() == ref.tx_ids).all())
nodes = nodes.set_index("tx_id").loc[ref.tx_ids].reset_index()
scaled = scaled.set_index("tx_id").loc[ref.tx_ids].reset_index()
ref_df = pd.DataFrame(ref.X_raw.astype(np.float64), columns=FEATS)
fit29 = (ref.time_steps <= 29) & (ref.y >= 0)  # the Spark scaler is fitted on labelled steps 1-29 (its train split)
ref_scaled = pd.DataFrame(PP.reference_scaled(ref.X_raw, fit29), columns=FEATS)
checks = {
    "node count (Spark / reference)": f"{len(nodes)} / {ref.n}",
    "node order identical": order_identical,
    "time steps identical": bool((nodes.time_step.astype(int).to_numpy() == ref.time_steps).all()),
    "labels identical": bool((nodes.label.to_numpy() == ref.y).all()),
    "edge count (Spark / reference)": f"{len(S['edges'])} / {ref.edge_index.shape[1]}",
}
e_sp = S["edges"]
idx_to_tx = pd.Series(S["nodes"].tx_id.to_numpy(), index=S["nodes"].node_idx.to_numpy())
sp_pairs = set(zip(idx_to_tx.loc[e_sp.source_idx].to_numpy(), idx_to_tx.loc[e_sp.target_idx].to_numpy()))
ref_pairs = set(zip(ref.tx_ids[ref.edge_index[0]], ref.tx_ids[ref.edge_index[1]]))
checks["edge sets identical"] = sp_pairs == ref_pairs
display(pd.Series(checks, name="value").to_frame())

# %% [markdown]
# ## 2. Feature-level error vs the reference
# * **raw features**: Spark parses the CSV as double and stores float32; the reference parses with pandas.
# * **scaled features**: Spark standardises in its own job (mean / population std of labelled steps 1–29 in
#   double precision, stored as float32); the reference recomputes the same in float64 NumPy.
#
# Definitions: RMSE, MAE, max |err|; relative error = max |err| / max |ref|; R² = 1 − SSE/SST; exact-match rate =
# share of bit-identical values; CCC = Lin's concordance correlation coefficient.

# %%
err_raw = PP.agreement_table(ref_df, nodes[FEATS].astype(np.float64), FEATS)
err_scaled = PP.agreement_table(ref_scaled, scaled[FEATS].astype(np.float64), FEATS)
summ = pd.concat({"raw (165)": err_raw.describe().T[["mean", "min", "max"]],
                  "scaled (165)": err_scaled.describe().T[["mean", "min", "max"]]}, axis=1)
PL.save_table(err_raw, "01_feature_error_raw", "Spark vs reference, raw features (per feature)", float_fmt="{:.3e}", png=False)
PL.save_table(err_scaled, "01_feature_error_scaled", "Spark vs reference, standardised features (per feature)",
              float_fmt="{:.3e}", png=False)
PL.save_table(summ, "01_feature_error_summary", "Spark vs reference: per-feature error, summarised over 165 features",
              float_fmt="{:.3e}")
display(summ)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1]})
for ax, (name, E_) in zip(axes, (("raw", err_raw), ("standardised", err_scaled))):
    M = E_[["RMSE", "MAE", "max_abs_err", "rel_err"]].T
    L = np.log10(M.clip(lower=1e-17))
    im = ax.imshow(L, aspect="auto", cmap=PL.SEQ_CMAP, vmin=-17, vmax=max(-5, float(L.to_numpy().max())))
    ax.set_yticks(range(4)); ax.set_yticklabels(["RMSE", "MAE", "max |err|", "relative err"])
    ax.set_xticks(range(0, 165, 10)); ax.set_xticklabels([FEATS[i] for i in range(0, 165, 10)], rotation=90, fontsize=6)
    ax.axvline(92.5, color=PL.INK2, lw=0.8, ls="--"); ax.grid(False)
    ax.set_title(f"{name}: log10 error per feature (dashed: local | aggregated); exact match "
                 f"{E_.exact_match_rate.mean():.2%}", loc="left", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="log10 (0 error shown at -17)")
fig.tight_layout(); PL.savefig(fig, "01_error_heatmap"); plt.show()

# %% [markdown]
# ## 3. Integrity: nulls, malformed lines, dropped rows

# %%
integ = PP.raw_csv_integrity(RAW)
vr = json.load(open(os.path.join(SPARK_DIR, "validation_report.json")))
integ_tab = pd.DataFrame({
    "raw CSV check (independent)": {
        "feature lines": integ["feature_lines"], "malformed lines (≠167 fields)": integ["malformed_lines"],
        "non-numeric lines": integ["non_numeric_lines"], "malformed-line rate": integ["malformed_lines"] / integ["feature_lines"],
        "duplicate tx ids": integ["duplicate_tx_ids"], "invalid labels": integ["invalid_labels"],
        "edges": integ["edge_rows"], "dangling edges (endpoint missing)": integ["dangling_edges"],
        "duplicate edges": integ["duplicate_edges"], "self-loops": integ["self_loops"],
        "time-step range": f"{integ['time_step_min']}-{integ['time_step_max']}"},
    "Spark pipeline report": {
        "feature lines": vr["raw_counts"]["features"], "malformed lines (≠167 fields)": "FAILFAST parse: 0 (job would abort)",
        "non-numeric lines": vr["integrity"]["non_finite_feature_values"], "malformed-line rate": 0.0,
        "duplicate tx ids": vr["integrity"]["duplicate_tx_ids_features"], "invalid labels": vr["integrity"]["invalid_labels"],
        "edges": vr["raw_counts"]["edges"], "dangling edges (endpoint missing)": vr["integrity"]["dangling_edge_endpoints"],
        "duplicate edges": vr["integrity"]["duplicate_edges"], "self-loops": vr["integrity"]["self_loops"],
        "time-step range": f"{vr['integrity']['time_step_min']}-{vr['integrity']['time_step_max']}"}})
null_rate = float(nodes[FEATS].isna().to_numpy().mean())
integ_tab.loc["null rate (output features)", :] = [float(np.isnan(ref.X_raw).mean()), null_rate]
integ_tab.loc["rows dropped (input → output)", :] = [integ["feature_lines"] - ref.n, vr["raw_counts"]["features"] - len(nodes)]
integ_tab.loc["cross-time-step edges", :] = [cross_step_edges(ref), vr["integrity"]["cross_time_step_edges"]]
PL.save_table(integ_tab.astype(str), "01_integrity", "Integrity counts: independent raw-CSV check vs Spark report")
display(integ_tab)

# %% [markdown]
# ## 4. Data quality before vs after preprocessing
# Computed on labelled **training** rows (steps 1–34). Three versions of the 165 features: raw, standardised
# (the pipeline's output) and asinh(standardised) (an optional heavy-tail transform, shown for comparison only;
# TGAT-BoostNet uses the standardised features).
#
# **Class SNR** of a feature = |μ_illicit − μ_licit| / √((σ²_illicit + σ²_licit)/2) (standardised mean difference).
# Standardisation is affine, so skewness, kurtosis, |z|-outliers and class SNR are *mathematically unchanged* by it —
# the table shows this explicitly; what standardisation changes is scale (mean 0, sd 1), which the GAT needs.

# %%
tr34 = ref.train_mask
Xs, mu, sd = standardise(ref)
versions = {"raw": ref.X_raw[tr34], "standardised": Xs[tr34], "asinh(standardised)": PP.signed_log(Xs[tr34])}
prof = {k: PP.quality_profile(v, FEATS) for k, v in versions.items()}
snr = {k: PP.class_snr(v, ref.y[tr34]) for k, v in versions.items()}
qrows = {}
for k, pdf in prof.items():
    qrows[k] = {"missing rate (mean)": pdf.missing_rate.mean(), "outliers |z|>3 (mean frac)": pdf.outlier_z3.mean(),
                "outliers 1.5 IQR (mean frac)": pdf.outlier_iqr.mean(), "|skewness| (median)": pdf.skewness.abs().median(),
                "excess kurtosis (median)": pdf.excess_kurtosis.median(), "class SNR (median)": np.median(snr[k]),
                "class SNR (max)": np.max(snr[k]), "feature mean (median |.|)": np.median(np.abs(versions[k].mean(0))),
                "feature sd (median)": np.median(versions[k].std(0)), "value range (median)": np.median(np.ptp(versions[k], 0))}
qtab = pd.DataFrame(qrows)
PL.save_table(qtab, "01_quality_before_after", "Data quality of the 165 features (training rows)", float_fmt="{:.4f}")
display(qtab)
full_q = pd.concat({k: pd.concat([p, pd.Series(snr[k], index=FEATS, name="class_snr")], axis=1) for k, p in prof.items()}, axis=1)
PL.save_table(full_q, "01_quality_per_feature", "Per-feature quality profile", png=False)

# %%
top_skew = prof["raw"].skewness.abs().sort_values(ascending=False).index[:3].tolist()
top_snr = pd.Series(snr["raw"], index=FEATS).sort_values(ascending=False).index[:3].tolist()
show = top_skew + top_snr
fig, axes = plt.subplots(len(show), 3, figsize=(11, 1.55 * len(show)))
yy = ref.y[tr34]
for i, f_ in enumerate(show):
    j = FEATS.index(f_)
    for c, (name, V) in enumerate(versions.items()):
        ax = axes[i, c]
        for cls, col, lab in ((0, PL.CATEGORICAL[0], "licit"), (1, PL.OURS_COLOR, "illicit")):
            v = V[yy == cls, j]
            lo, hi = np.percentile(V[:, j], [0.5, 99.5])
            ax.hist(np.clip(v, lo, hi), bins=60, density=True, histtype="stepfilled", alpha=0.45, color=col, label=lab)
        ax.set_yticks([])
        if i == 0:
            ax.set_title(name)
        if c == 0:
            ax.set_ylabel(f"{f_}\n({'high skew' if f_ in top_skew else 'high SNR'})", fontsize=7)
        ax.text(0.98, 0.85, f"skew {prof[name].skewness[f_]:.1f}\nSNR {snr[name][j]:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=PL.INK2)
axes[0, 0].legend(loc="upper left", fontsize=6.5)
fig.suptitle("Before / after distributions (clipped to the 0.5-99.5 % range for display)", x=0.01, ha="left", fontsize=9)
fig.tight_layout(); PL.savefig(fig, "01_before_after_distributions"); plt.show()

# %%
bal = []
for name, (lo, hi) in {"train 1-34": (1, 34), "test 35-49": (35, 49), "all labelled": (1, 49)}.items():
    m = ref.mask((lo, hi))
    n1, n0 = int((ref.y[m] == 1).sum()), int((ref.y[m] == 0).sum())
    bal.append({"split": name, "labelled": n1 + n0, "illicit": n1, "licit": n0, "illicit %": 100 * n1 / (n1 + n0),
                "imbalance licit:illicit": n0 / n1})
unl = int((ref.y == -1).sum())
bal = pd.DataFrame(bal).set_index("split")
bal.loc["unknown label (kept in graph, never in loss/metric)", "labelled"] = unl
PL.save_table(bal, "01_class_balance", "Class balance per split", float_fmt="{:.2f}")
display(bal)
print(f"Training licit:illicit ratio = {bal.loc['train 1-34', 'imbalance licit:illicit']:.2f}; TGAT-BoostNet's fixed "
      f"cost weight w_illicit = 4.32 is applied on top of that.")

# %% [markdown]
# ## 5. Leakage / causality checks
# 1. **Train-only scaler.** The Spark scaler statistics must equal statistics of the training rows and differ from
#    statistics that include test rows.
# 2. **Future-perturbation test.** Scramble every feature of time steps 35–49 (×U(0.1, 10) plus noise), re-run the
#    preprocessing, and compare the features of steps 1–34. A causal pipeline leaves them *bit-identical*; a
#    non-causal control (scaler fitted on all rows) does not.
# 3. **Graph causality.** Elliptic edges never cross time steps, so a GNN's receptive field stays inside one step.
#    We verify it with the trained GAT: perturb all test-period features and check that the embeddings of every
#    training-period node are unchanged.

# %%
sstats = json.load(open(os.path.join(SPARK_DIR, "scaler_stats.json")))
mu29 = ref.X_raw[fit29].astype(np.float64).mean(0); sd29 = ref.X_raw[fit29].astype(np.float64).std(0)
mu_all = ref.X_raw[ref.y >= 0].astype(np.float64).mean(0)
leak1 = pd.Series({"max |Spark mean - train(1-29) mean|": np.abs(np.array(sstats["mean"]) - mu29).max(),
                   "max |Spark std - train(1-29) std|": np.abs(np.array(sstats["std_population"]) - sd29).max(),
                   "max |Spark mean - all-labelled mean| (would be 0 if test rows leaked)": np.abs(np.array(sstats["mean"]) - mu_all).max()})
display(leak1.to_frame("value"))
rng = np.random.default_rng(0)
Xp = ref.X_raw.copy()
fut = ref.time_steps >= 35
Xp[fut] = Xp[fut] * rng.uniform(0.1, 10, size=Xp[fut].shape).astype(np.float32) + rng.normal(0, 1, Xp[fut].shape).astype(np.float32)
causal_a = standardise(ref)[0]; causal_b = standardise(ref, Xp)[0]   # scaler fitted on labelled steps 1-34 only
def allfit(X):
    m = X[ref.y >= 0].astype(np.float64).mean(0); s = X[ref.y >= 0].astype(np.float64).std(0); s[s == 0] = 1
    return ((X - m) / s).astype(np.float32)
leaky_a, leaky_b = allfit(ref.X_raw), allfit(Xp)
past = ref.time_steps <= 34
d_causal = np.abs(causal_a[past] - causal_b[past]).max(0)
d_leaky = np.abs(leaky_a[past] - leaky_b[past]).max(0)
per_step = pd.DataFrame({"causal (train-only scaler)": [np.abs(causal_a[ref.time_steps == t] - causal_b[ref.time_steps == t]).max() for t in range(1, 50)],
                         "non-causal control (all-row scaler)": [np.abs(leaky_a[ref.time_steps == t] - leaky_b[ref.time_steps == t]).max() for t in range(1, 50)]},
                        index=range(1, 50))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for col, c, ls in (("causal (train-only scaler)", PL.OURS_COLOR, "-"), ("non-causal control (all-row scaler)", PL.CATEGORICAL[0], "--")):
    axes[0].plot(per_step.index, per_step[col].clip(lower=1e-12), color=c, ls=ls, marker="o", ms=3, label=col)
axes[0].axvspan(34.5, 49.5, color=PL.GREY, alpha=0.15, lw=0); axes[0].text(42, 2e-12, "perturbed steps", ha="center", fontsize=7)
axes[0].set_yscale("log"); axes[0].set(xlabel="time step", ylabel="max |change| of preprocessed features",
                                         title="Future-perturbation test (1e-12 = exactly 0)"); axes[0].legend(loc="center left")
axes[1].bar(np.arange(165), d_leaky, color=PL.CATEGORICAL[0], width=1, label="non-causal control")
axes[1].bar(np.arange(165), d_causal, color=PL.OURS_COLOR, width=1, label=f"causal: max {d_causal.max():.1e}")
axes[1].set(xlabel="feature index", ylabel="max |change| on steps 1-34", title="Per-feature change of past rows")
axes[1].legend(); fig.tight_layout(); PL.savefig(fig, "01_leakage_future_perturbation"); plt.show()
print(f"causal pipeline: steps 1-34 unchanged = {bool(d_causal.max() == 0)} | non-causal control max change = {d_leaky.max():.3f}")

# %%
from src.evaluation import baselines as B
from src.evaluation.data import bidirectional
art = os.path.join(B.RESULTS_DIR, "artifacts", f"{B.OURS}_seed42", "gat.pt")
if os.path.exists(art):
    from src.models.graph_encoder import load_encoder
    gat = load_encoder(torch.load(art, map_location="cpu"))
    E = torch.as_tensor(bidirectional(ref.edge_index))
    with torch.no_grad():
        e_a = gat.embed(torch.as_tensor(causal_a), E).numpy()
        e_b = gat.embed(torch.as_tensor(causal_b), E).numpy()   # causal_b = preprocessed perturbed data
    g_change = np.abs(e_a[past] - e_b[past]).max()
    print(f"cross-step edges = {cross_step_edges(ref)}; max change of GAT embeddings of steps 1-34 after perturbing "
          f"steps 35-49 (full-graph forward) = {g_change:.3g}")
else:
    print("GAT artifact not found (run notebook 02 first); graph-causality test skipped")

# %% [markdown]
# ## 6. Efficiency: runtime, throughput, memory, disk, scaling
# The Spark job is run on time steps 1–12, 1–24, 1–36 and 1–49 (CSV rows filtered by step; edges are intra-step).
# Memory is sampled every 50 ms with the repository's tracker (driver RSS, JVM child RSS, whole process tree).
# Single machine, `local[4]`: this is a workstation measurement, not a cluster benchmark.

# %%
bench = cached_json("elliptic_spark_scaling", lambda: [PP.benchmark_elliptic_spark(RAW, k) for k in (12, 24, 36, 49)], RUN_SPARK)
refb = cached_json("pandas_reference_timing", lambda: PP.benchmark_pandas_reference(RAW), RUN_SPARK)
bdf = pd.DataFrame(bench).set_index("max_step")
PL.save_table(bdf, "01_spark_scaling", "Elliptic Spark preprocessing over growing step subsets", float_fmt="{:.2f}")
display(bdf); display(pd.Series(refb, name="pandas reference (full data)").to_frame())
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].plot(bdf.n_nodes, bdf.wall_seconds, "o-", color=PL.OURS_COLOR, label="Spark wall time (incl. session start)")
axes[0].plot(bdf.n_nodes, bdf.pipeline_seconds, "s--", color=PL.CATEGORICAL[0], label="Spark job (excl. session start)")
axes[0].set(xlabel="transactions", ylabel="seconds", title="Runtime"); axes[0].legend()
axes[1].plot(bdf.n_nodes, bdf.tx_per_s, "o-", color=PL.OURS_COLOR, label="transactions / s")
axes[1].plot(bdf.n_nodes, bdf.lines_per_s, "s--", color=PL.CATEGORICAL[0], label="input lines / s")
axes[1].set(xlabel="transactions", ylabel="per second", title="Throughput"); axes[1].legend()
for col, c, ls in (("tree_peak_rss_mb", PL.OURS_COLOR, "-"), ("jvm_peak_rss_mb", PL.CATEGORICAL[0], "--"), ("driver_peak_rss_mb", PL.CATEGORICAL[1], "-.")):
    axes[2].plot(bdf.n_nodes, bdf[col], marker="o", ls=ls, color=c, label=col.replace("_peak_rss_mb", " peak RSS"))
axes[2].set(xlabel="transactions", ylabel="MB", title="Peak memory"); axes[2].legend()
fig.tight_layout(); PL.savefig(fig, "01_spark_scaling_curve"); plt.show()
disk = pd.Series({"input CSVs (MB)": bdf.loc[49, "input_mb"], "Spark Parquet output (MB)": bdf.loc[49, "output_mb"],
                  "compression ratio": bdf.loc[49, "input_mb"] / bdf.loc[49, "output_mb"]})
display(disk.to_frame("full data"))

# %% [markdown]
# ## 7. Downstream effect: raw vs preprocessed features
# Same models, same seeds, same split; only the input preprocessing differs. Tree ensembles are invariant to
# monotone per-feature scaling (identical splits), so the effect is shown for scale-sensitive models: logistic
# regression, MLP and TGAT-BoostNet (raw vs standardised GAT input; ablation run `abl_rawfeat`).

# %%
from src.evaluation.metrics import full_metrics
y_te = ref.y[ref.test_mask]
ds = []
for key in ("lr", "mlp"):
    for prep, Xin in (("raw", ref.X_raw), ("standardised", None)):
        for s in (B.seeds_for(key) if not QUICK else [42]):
            r = B.run_tabular(key, ref, s, DEVICE, X=Xin) if prep == "raw" else B.run(key, ref, s, DEVICE)
            m = full_metrics(y_te, r["p_test"])
            ds.append(dict(model=B.MODELS[key][0], prep=prep, seed=s, f1=m["f1"], roc_auc=m["roc_auc"], pr_auc=m["pr_auc"]))
for key, prep in ((B.OURS, "standardised"), ("abl_rawfeat", "raw")):
    for s in SEEDS:
        if B.has_run(key, s) or not QUICK:
            m = full_metrics(y_te, B.run(key, ref, s, DEVICE)["p_test"])
            ds.append(dict(model="TGAT-BoostNet", prep=prep, seed=s, f1=m["f1"], roc_auc=m["roc_auc"], pr_auc=m["pr_auc"]))
ds = pd.DataFrame(ds)
dtab = ds.groupby(["model", "prep"])[["f1", "roc_auc", "pr_auc"]].agg(["mean", "std"])
dtab.columns = [f"{a}_{b}" for a, b in dtab.columns]
PL.save_table(dtab, "01_downstream_raw_vs_preprocessed", "Classifier performance, raw vs standardised features (test 35-49)")
display(dtab)
fig, axes = plt.subplots(1, 3, figsize=(12, 2.8), sharey=True)
mods = list(dict.fromkeys(ds.model))
for ax, met in zip(axes, ("f1", "roc_auc", "pr_auc")):
    for i, mname in enumerate(mods):
        a = ds[(ds.model == mname) & (ds.prep == "raw")][met].mean()
        b = ds[(ds.model == mname) & (ds.prep == "standardised")][met].mean()
        ax.plot([a, b], [i, i], color=PL.GREY, lw=2, zorder=1)
        ax.scatter([a], [i], color=PL.CATEGORICAL[0], s=40, zorder=3, label="raw" if i == 0 else None)
        ax.scatter([b], [i], color=PL.OURS_COLOR, s=40, zorder=3, label="standardised" if i == 0 else None)
    ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods); ax.set_xlabel(met.upper().replace("_", "-"))
axes[0].legend(loc="lower right")
fig.suptitle("Downstream effect of preprocessing (mean of seeds)", x=0.01, ha="left", fontsize=9)
fig.tight_layout(); PL.savefig(fig, "01_downstream_dumbbell"); plt.show()

# %% [markdown]
# ## A. Appendix — Zenodo Spark pipeline vs the pure-Python causal reference
# A 100,000-line raw slice of Zenodo part `aa` (streamed from Zenodo record 7157854 if not on disk). The Spark
# pipeline and the independent reference (`compute_ground_truth_for_keys`) both compute all **15 downstream
# features** for every completed transaction of the slice.

# %%
ZEN_LINES = 20000 if QUICK else 100000
slice_path = os.path.join(PRE_DIR, f"zenodo_slice_{ZEN_LINES}.txt")
if not os.path.exists(slice_path):
    PP.zenodo_slice(ZEN_LINES, slice_path)
zcache = os.path.join(PRE_DIR, f"zenodo_spark_{ZEN_LINES}.parquet")
if RUN_SPARK or not os.path.exists(zcache):
    zsp, zcounts = PP.zenodo_spark(slice_path)
    zsp.to_parquet(zcache); json.dump(zcounts, open(zcache + ".json", "w"), indent=1)
zsp = pd.read_parquet(zcache); zcounts = json.load(open(zcache + ".json"))
t0 = time.perf_counter(); zref = PP.zenodo_reference(slice_path); t_ref = time.perf_counter() - t0
key = ["block_number", "tx_index"]
mz = zref.merge(zsp, on=key, suffixes=("_ref", "_spark"), how="outer", indicator=True)
print("rows: reference", len(zref), "| Spark", len(zsp), "| matched", int((mz._merge == "both").sum()))
mz = mz[mz._merge == "both"]
zerr = PP.agreement_table(mz[[f + "_ref" for f in PP.ZENODO_FEATURES]].set_axis(PP.ZENODO_FEATURES, axis=1),
                          mz[[f + "_spark" for f in PP.ZENODO_FEATURES]].set_axis(PP.ZENODO_FEATURES, axis=1), PP.ZENODO_FEATURES)
PL.save_table(zerr, "01A_zenodo_feature_error", f"Zenodo: Spark vs pure-Python reference, {ZEN_LINES:,}-line slice",
              float_fmt="{:.3e}")
display(zerr)
fig, ax = plt.subplots(figsize=(8, 4.2))
L = np.log10(zerr[["RMSE", "MAE", "max_abs_err", "rel_err"]].clip(lower=1e-17))
im = ax.imshow(L.T, aspect="auto", cmap=PL.SEQ_CMAP, vmin=-17, vmax=max(-5, float(L.to_numpy().max())))
ax.set_xticks(range(15)); ax.set_xticklabels(PP.ZENODO_FEATURES, rotation=50, ha="right", fontsize=7)
ax.set_yticks(range(4)); ax.set_yticklabels(["RMSE", "MAE", "max |err|", "relative err"]); ax.grid(False)
for i in range(4):
    for j in range(15):
        v = zerr[["RMSE", "MAE", "max_abs_err", "rel_err"]].iat[j, i]
        ax.text(j, i, "0" if v == 0 else f"{v:.0e}", ha="center", va="center", fontsize=6, color=PL.INK)
fig.colorbar(im, ax=ax, fraction=0.03, label="log10 error (0 shown at -17)")
ax.set_title("Zenodo: per-feature error of Spark vs reference", loc="left"); fig.tight_layout()
PL.savefig(fig, "01A_zenodo_error_heatmap"); plt.show()

# %%
zq = pd.Series({**zcounts, "malformed-line rate": zcounts["malformed_lines"] / zcounts["raw_lines"],
                "dropped boundary/incomplete tx rate": zcounts["dropped_incomplete_tx"] / zcounts["tx_groups"],
                "null rate (15 features)": float(zsp[PP.ZENODO_FEATURES].isna().to_numpy().mean()),
                "reference seconds (pure Python)": t_ref, **{f"sanity: {k}": v for k, v in PP.zenodo_sanity(zsp).items()}})
PL.save_table(zq.to_frame("value"), "01A_zenodo_integrity", "Zenodo slice: counts, rates and fee/volume sanity", float_fmt="{:.4g}")
display(zq.to_frame("value"))
caus = PP.zenodo_causality_test(slice_path)
print(f"cut at line {caus.attrs['cut_line']:,} (key {caus.attrs['cut_key']}); {caus.attrs['n_pre_cut_tx']:,} transactions before the cut")
PL.save_table(caus, "01A_zenodo_causality", "Zenodo causality: max change of pre-cut features after scrambling all later lines",
              float_fmt="{:.4g}")
display(caus)
fig, ax = plt.subplots(figsize=(8, 3.4))
vals = caus["causal_reference_max_abs_change"].dropna()
ax.bar(range(len(vals)), vals.clip(lower=1e-3), color=PL.OURS_COLOR, label="causal reference (0 = unchanged)")
nc = caus["noncausal_control_max_abs_change"].dropna().iloc[0]
ax.bar([len(vals) + 1], [nc], color=PL.CATEGORICAL[0], label="non-causal control (total history)")
ax.set_xticks(list(range(len(vals))) + [len(vals) + 1]); ax.set_xticklabels(list(vals.index) + ["total tx count\n(control)"], rotation=60, ha="right", fontsize=7)
ax.set_yscale("log"); ax.set_ylabel("max |change| (1e-3 = exactly 0)"); ax.legend(loc="upper left")
ax.set_title("Zenodo: features of past transactions after perturbing the future", loc="left"); fig.tight_layout()
PL.savefig(fig, "01A_zenodo_causality"); plt.show()

# %% [markdown]
# ### A.2 Zenodo scaling curve (existing reports, loaded from JSON — not retyped)

# %%
sc = PP.load_scale_reports()
PL.save_table(sc, "01A_zenodo_scaling", "Zenodo Spark preprocessing: bounded slices (v3 reports) and the full run",
              float_fmt="{:.2f}")
display(sc)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].loglog(sc.raw_lines, sc.wall_seconds, "o-", color=PL.OURS_COLOR); axes[0].set(xlabel="raw lines", ylabel="seconds", title="Runtime")
axes[1].semilogx(sc.raw_lines, sc.lines_per_s, "o-", color=PL.OURS_COLOR, label="lines / s")
axes[1].semilogx(sc.raw_lines, sc.tx_per_s, "s--", color=PL.CATEGORICAL[0], label="transactions / s")
axes[1].set(xlabel="raw lines", title="Throughput"); axes[1].legend()
for col, c, ls in (("tree_peak_rss_mb", PL.OURS_COLOR, "-"), ("jvm_peak_rss_mb", PL.CATEGORICAL[0], "--"), ("driver_peak_rss_mb", PL.CATEGORICAL[1], "-.")):
    axes[2].semilogx(sc.raw_lines, sc[col], marker="o", ls=ls, color=c, label=col.replace("_peak_rss_mb", ""))
axes[2].set(xlabel="raw lines", ylabel="peak RSS (MB)", title="Memory"); axes[2].legend()
for ax in axes:
    for x, lab in zip(sc.raw_lines, sc.index):
        ax.axvline(x, color=PL.GREY, lw=0.4, ls=":")
fig.tight_layout(); PL.savefig(fig, "01A_zenodo_scaling_curve"); plt.show()

# %% [markdown]
# ## Summary

# %%
txt = (f"**Elliptic (feeds TGAT-BoostNet).** Spark vs reference: raw features exact-match rate "
       f"{err_raw.exact_match_rate.mean():.2%}, max |err| {err_raw.max_abs_err.max():.2e}; standardised features "
       f"max |err| {err_scaled.max_abs_err.max():.2e} (float32 storage), min R² {err_scaled.R2.min():.6f}. "
       f"Null rate {null_rate:.0%}, malformed lines {integ['malformed_lines']}, dangling edges {integ['dangling_edges']}. "
       f"Future-perturbation test: past features unchanged = {bool(d_causal.max() == 0)}. "
       f"Spark runtime on all 49 steps: {bdf.loc[49, 'wall_seconds']:.0f} s ({bdf.loc[49, 'tx_per_s']:.0f} tx/s), "
       f"peak process-tree RSS {bdf.loc[49, 'tree_peak_rss_mb']:.0f} MB.  \n"
       f"**Zenodo appendix.** {len(mz):,} transactions compared on 15 features; max |err| {zerr.max_abs_err.max():.2e}, "
       f"exact-match rate {zerr.exact_match_rate.mean():.2%}; malformed lines {zcounts['malformed_lines']}, "
       f"dropped incomplete boundary tx {zcounts['dropped_incomplete_tx']}.")
display(Markdown(txt))
open(os.path.join(PL.TAB_DIR, "01_summary.md"), "w", encoding="utf-8").write(txt + "\n")

# %% [markdown]
# ## Download figures and tables

# %%
import zipfile
zp = os.path.join(ROOT, "01_preprocessing_metrics_outputs.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in (PL.FIG_DIR, PL.TAB_DIR):
        for f in sorted(os.listdir(d)):
            if f.startswith("01"):
                zf.write(os.path.join(d, f), os.path.join(os.path.basename(d), f))
print("wrote", zp)
if IN_COLAB:
    from google.colab import files
    files.download(zp)
