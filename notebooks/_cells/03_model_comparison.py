# %% [markdown]
# # 03 · TGAT-BoostNet vs existing models
# **Course 21CSC314P (SRMIST) · Distributed Streaming Anomaly Detection on Blockchain Transaction Graphs**
#
# Two kinds of numbers appear here and are **never mixed**:
#
# | Label | Meaning |
# |---|---|
# | **reproduced** | trained and scored by this notebook (or the cached runs it reads) on the identical split, seeds and threshold |
# | **reported in paper** | transcribed from the paper's table, with citation; not re-run, not used for ranking |
#
# Protocol for every reproduced model: train on labelled steps 1–34, test on labelled steps 35–49, illicit = positive,
# threshold 0.5, hyper-parameters fixed a priori (published configuration where one exists, otherwise library
# defaults). Stochastic models: 5 seeds (42, 123, 456, 789, 2024); deterministic ones (logistic regression): 1.
#
# **Runtime:** ~10 min with the cached runs in `results/`; from scratch ~1.5 h on a Colab T4, ~8 h on a 10-core CPU.
# The final cell writes an automatic, honest summary of where TGAT-BoostNet ranks.

# %% INCLUDE_SETUP

# %% [markdown]
# ## 1. Data

# %%
from src.evaluation.data import load_elliptic, SEEDS, standardise
from src.evaluation import baselines as B, stats as ST
from src.evaluation.metrics import full_metrics, METRIC_LABELS, METRIC_DIRECTION, per_timestep, fast_counts_metrics
from src.evaluation.literature import reported_table, SOURCES
data = load_elliptic()
tr, te = data.train_mask, data.test_mask
y_te, steps_te = data.y[te], data.time_steps[te]
OURS = B.OURS
LAB = {k: v[0] for k, v in B.MODELS.items()}
N_BOOT = 300 if QUICK else 2000
print(f"train {tr.sum()} labelled ({(data.y[tr] == 1).sum()} illicit) | test {te.sum()} ({(y_te == 1).sum()} illicit)")

# %% [markdown]
# ## 2. Published results — *reported in paper*
# Transcribed from each paper's own table (same split: train 1–34 / test 35–49). See `src/evaluation/literature.py`
# for the exact table numbers and caveats.

# %%
lit = reported_table()
lit_show = lit[["model", "features", "precision", "recall", "f1_illicit", "micro_f1", "auc", "seeds", "source", "year",
                "status"]].set_index(["source", "model", "features"])
PL.save_table(lit_show, "03_reported_in_paper", "Published Elliptic results (REPORTED IN PAPER; split 1-34 / 35-49)")
display(lit_show)
for k, v in SOURCES.items():
    print(f"[{k}] {v}")

# %% [markdown]
# ## 3. Reproduced models on the identical split
# `B.collect` loads each cached run or trains it. Models: logistic regression, SVM, random forest, XGBoost,
# LightGBM, CatBoost, MLP, GCN, Skip-GCN, GraphSAGE, GAT alone, EvolveGCN-O, GAT+RF, GCN+XGBoost, GBDT-165 and
# GBDT-197 without cost weighting.

# %%
keys = list(B.COMPARISON)
try:
    import catboost  # noqa: F401
except ImportError:
    keys.remove("catboost"); print("catboost not installed -> CatBoost skipped")
runs = B.collect(keys, data, DEVICE)
P = {k: [runs[k][s]["p_test"] for s in sorted(runs[k])] for k in keys}

def metric_frame(k):
    return pd.DataFrame([full_metrics(y_te, p) for p in P[k]])

MF = {k: metric_frame(k) for k in keys}
SHOW = ["f1", "precision", "recall", "f2", "mcc", "balanced_accuracy", "roc_auc", "pr_auc", "brier", "ece"]
mean = pd.DataFrame({LAB[k]: MF[k][list(METRIC_LABELS)].mean() for k in keys}).T
std = pd.DataFrame({LAB[k]: MF[k][list(METRIC_LABELS)].std(ddof=1) for k in keys}).T.fillna(0)
mean = mean.sort_values("f1", ascending=False)
std = std.loc[mean.index]
board = pd.DataFrame({METRIC_LABELS[c]: [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c])] for c in SHOW},
                     index=mean.index)
board.insert(0, "seeds", [len(P[k]) for k in [next(kk for kk in keys if LAB[kk] == lab) for lab in mean.index]])
board.insert(0, "status", "reproduced")
PL.save_table(board, "03_leaderboard", "Reproduced models, test steps 35-49, threshold 0.5 (mean ± std over seeds)",
              highlight_index=LAB[OURS])
PL.save_table(mean, "03_leaderboard_all_metrics_mean", "All metrics, mean over seeds", highlight_index=LAB[OURS])
PL.save_table(std, "03_leaderboard_all_metrics_std", "All metrics, std over seeds", highlight_index=LAB[OURS])
display(board)

# %% [markdown]
# ## 4. Reproduced vs reported, side by side
# For every published model we also re-ran, both numbers are shown in separate, labelled columns.

# %%
rr = lit[lit.maps_to.notna() & lit.maps_to.isin(keys)].copy()
rr["reported F1 (paper)"] = rr.f1_illicit
rr["reproduced F1 (ours, mean ± std)"] = [f"{MF[k].f1.mean():.3f} ± {MF[k].f1.std(ddof=1) if len(MF[k]) > 1 else 0:.3f}"
                                         for k in rr.maps_to]
rr["reported AUC (paper)"] = rr.auc
rr["reproduced AUC"] = [round(MF[k].roc_auc.mean(), 3) for k in rr.maps_to]
rr["reproduction config"] = [LAB[k] for k in rr.maps_to]
rr = rr.set_index(["model", "source"])[["features", "reported F1 (paper)", "reproduced F1 (ours, mean ± std)",
                                        "reported AUC (paper)", "reproduced AUC", "reproduction config"]]
PL.save_table(rr, "03_reproduced_vs_reported", "Reproduced (this work) vs reported in paper")
display(rr)

# %% [markdown]
# ## 5. Leaderboard heatmap, radar chart, ROC/PR overlay

# %%
featured = [k for k in [next(kk for kk in keys if LAB[kk] == lab) for lab in mean.index] if k != OURS][:5]
ranks = ST.rank_table(mean[SHOW], METRIC_DIRECTION)
hm_vals = mean[SHOW].copy(); hm_vals.columns = [METRIC_LABELS[c] for c in SHOW]
hm_rank = ranks.copy(); hm_rank.columns = hm_vals.columns
fig = PL.heatmap_leaderboard(hm_vals, hm_rank, LAB[OURS], "Leaderboard: value and rank per metric (1 = best)")
PL.savefig(fig, "03_heatmap_leaderboard"); plt.show()
RAD = ["f1", "precision", "recall", "mcc", "roc_auc", "pr_auc", "balanced_accuracy"]
rad = mean[RAD].copy()
rad["1 - Brier"] = 1 - mean["brier"]; rad["1 - ECE"] = 1 - mean["ece"]
rad.columns = [METRIC_LABELS.get(c, c) for c in rad.columns]
rad.index = [next(kk for kk in keys if LAB[kk] == lab) for lab in rad.index]
fig = PL.radar(rad, OURS, featured, LAB); PL.savefig(fig, "03_radar"); plt.show()
fig = PL.roc_pr_overlay(y_te, P, LAB, OURS, featured); PL.savefig(fig, "03_roc_pr_overlay"); plt.show()

# %% [markdown]
# ## 6. Seed variability (raincloud) and per-time-step F1

# %%
long = pd.concat([MF[k].assign(label=LAB[k], seed=sorted(runs[k])) for k in keys])
order = list(mean.index)
for metric, lab in (("f1", "Illicit F1 (threshold 0.5)"), ("pr_auc", "PR-AUC")):
    fig = PL.raincloud(long, metric, order, LAB[OURS], lab); PL.savefig(fig, f"03_raincloud_{metric}"); plt.show()
rows = []
for k in keys:
    for s, p in zip(sorted(runs[k]), P[k]):
        rows += [dict(r, model=k, seed=s) for r in per_timestep(y_te, p, steps_te)]
pts = pd.DataFrame(rows)
counts = pts[(pts.model == OURS) & (pts.seed == SEEDS[0])].set_index("time_step").n_illicit
fig = PL.per_timestep_lines(pts, OURS, featured[:4], LAB, counts=counts); PL.savefig(fig, "03_per_timestep_f1"); plt.show()
pt_mean = pts.groupby(["model", "time_step"]).f1.mean().unstack(0).rename(columns=LAB)
PL.save_table(pt_mean, "03_per_timestep_f1", "Per-time-step illicit F1 (mean over seeds)")

# %% [markdown]
# ## 7. Confusion matrices and calibration

# %%
cms = {}
for k in [next(kk for kk in keys if LAB[kk] == lab) for lab in mean.index]:
    m = MF[k].iloc[0]
    cms[k] = np.array([[m.tn, m.fp], [m.fn, m.tp]])
fig = PL.confusion_grid(cms, LAB, OURS, ncols=6); PL.savefig(fig, "03_confusion_grid"); plt.show()
fig = PL.calibration_and_threshold(y_te, {k: P[k][0] for k in [OURS] + featured[:4]}, LAB, OURS, featured)
PL.savefig(fig, "03_calibration"); plt.show()

# %% [markdown]
# ## 8. Statistical significance vs every baseline
# * **McNemar (exact)** on the hard predictions of the seed-averaged probability (threshold 0.5).
# * **Paired bootstrap** (2,000 resamples of test transactions) of the difference in 5-seed-mean F1 and PR-AUC.
# * p-values Holm-corrected across the baselines. Positive Δ = TGAT-BoostNet better.

# %%
from sklearn.metrics import average_precision_score
f1_fn = lambda y, p: fast_counts_metrics(y, (p >= 0.5).astype(int))["f1"]
ap_fn = lambda y, p: average_precision_score(y, p)
pm = {k: np.mean(P[k], axis=0) for k in keys}
sig = {}
for k in keys:
    if k == OURS:
        continue
    mc = ST.mcnemar(y_te, (pm[OURS] >= 0.5).astype(int), (pm[k] >= 0.5).astype(int))
    bf = ST.paired_bootstrap(y_te, P[OURS], P[k], f1_fn, n_boot=N_BOOT)
    ba = ST.paired_bootstrap(y_te, P[OURS], P[k], ap_fn, n_boot=max(N_BOOT // 4, 100))
    sig[LAB[k]] = {"ΔF1": bf["diff"], "ΔF1 95% CI": f"[{bf['ci_low']:+.3f}, {bf['ci_high']:+.3f}]", "p (F1 boot)": bf["p_value"],
                   "ΔPR-AUC": ba["diff"], "ΔPR-AUC 95% CI": f"[{ba['ci_low']:+.3f}, {ba['ci_high']:+.3f}]",
                   "p (AP boot)": ba["p_value"], "McNemar b (ours right)": mc["a_right_b_wrong"],
                   "McNemar c (baseline right)": mc["a_wrong_b_right"], "p (McNemar)": mc["p_value"]}
sig = pd.DataFrame(sig).T
for col in ("p (F1 boot)", "p (AP boot)", "p (McNemar)"):
    sig[col + " Holm"] = sig.index.map(ST.holm(sig[col].astype(float).to_dict()))
sig = sig.sort_values("ΔF1")
def verdict(r):
    if float(r["p (F1 boot) Holm"]) < 0.05:
        return "ours significantly better (F1)" if r["ΔF1"] > 0 else "baseline significantly better (F1)"
    return "no significant F1 difference"
sig["verdict"] = sig.apply(verdict, axis=1)
for c in sig.columns:
    if c.startswith("Δ") and "CI" not in c or c.startswith("p ("):
        sig[c] = sig[c].astype(float)
PL.save_table(sig, "03_significance", "TGAT-BoostNet vs each baseline: paired bootstrap and McNemar (Holm-corrected)",
              float_fmt="{:.4f}")
display(sig[["ΔF1", "ΔF1 95% CI", "p (F1 boot) Holm", "ΔPR-AUC", "ΔPR-AUC 95% CI", "p (AP boot) Holm",
             "p (McNemar) Holm", "verdict"]])

# %% [markdown]
# ## 9. Critical-difference diagram and rank summary
# Friedman test over the 15 test time steps (blocks; per-step F1, mean over seeds) and Nemenyi CD at α = 0.05.
# Models joined by a bar are not significantly different. Steps after 43 have F1 ≈ 0 for most models (ties).

# %%
blocks = pt_mean.copy()
fr = ST.friedman_nemenyi(blocks, higher_is_better=True)
print(f"Friedman chi2 = {fr['friedman_chi2']:.2f}, p = {fr['friedman_p']:.2e}, k = {fr['k']}, N = {fr['n_blocks']}, CD = {fr['cd']:.2f}")
fig = PL.cd_diagram(fr["avg_ranks"], fr["cd"], LAB[OURS], "Average rank of per-time-step F1 (Nemenyi CD, alpha=0.05)")
PL.savefig(fig, "03_critical_difference"); plt.show()
rank_all = ST.rank_table(mean[list(METRIC_LABELS)], METRIC_DIRECTION)
rank_all.columns = [METRIC_LABELS[c] for c in rank_all.columns]
rank_all["mean rank"] = rank_all.mean(1)
rank_all["time-step avg rank (Friedman)"] = fr["avg_ranks"].reindex(rank_all.index)
rank_all = rank_all.sort_values("mean rank")
PL.save_table(rank_all, "03_rank_summary", "Rank of each model on each metric (1 = best)", highlight_index=LAB[OURS],
              float_fmt="{:.2f}")
display(rank_all)

# %% [markdown]
# ## 10. Ablation study
# Each variant changes one component of TGAT-BoostNet and is run for the same 5 seeds:
# no GAT (GBDT on 165 features), no cost weighting (w = 1), 1/2/4/8 attention heads, embedding dim 16/32/64,
# + temporal feature (time step given to the GAT as t/49 and to the GBDT), and raw (unstandardised) GAT input.

# %%
abl_keys = ["gbdt165_w", "gbdt197_unw"] + list(B.ABLATIONS)
abl_runs = B.collect(abl_keys, data, DEVICE)
ab = {}
for k in [OURS] + abl_keys:
    src = runs[k] if k in runs else abl_runs[k]
    mf = pd.DataFrame([full_metrics(y_te, src[s]["p_test"]) for s in sorted(src)])
    ab[k] = {"F1": mf.f1.mean(), "F1 sd": mf.f1.std(ddof=1), "PR-AUC": mf.pr_auc.mean(), "PR-AUC sd": mf.pr_auc.std(ddof=1),
             "ROC-AUC": mf.roc_auc.mean(), "Precision": mf.precision.mean(), "Recall": mf.recall.mean(), "seeds": len(mf)}
names = {OURS: "TGAT-BoostNet (4 heads, dim 32, w=4.32)", "gbdt165_w": "no GAT (GBDT on 165)",
         "gbdt197_unw": "no cost weighting (w = 1)", "abl_heads1": "1 head", "abl_heads2": "2 heads",
         "abl_heads8": "8 heads", "abl_dim16": "embedding dim 16", "abl_dim64": "embedding dim 64",
         "abl_temporal": "+ temporal feature", "abl_rawfeat": "raw (unscaled) GAT input"}
ab = pd.DataFrame(ab).T.rename(index=names)
ref = ab.loc[names[OURS]]
ab["ΔF1 vs full"] = ab.F1 - ref.F1
ab["ΔPR-AUC vs full"] = ab["PR-AUC"] - ref["PR-AUC"]
PL.save_table(ab, "03_ablation", "Ablation of TGAT-BoostNet (5 seeds each)", highlight_index=names[OURS])
display(ab)
for metric, err in (("F1", "F1 sd"), ("PR-AUC", "PR-AUC sd")):
    d = ab.drop(index=names[OURS]).sort_values(metric)
    fig = PL.lollipop(d, metric, err, ref[metric], "full model", f"test {metric} (mean of 5 seeds; whisker = 1 sd)",
                      f"Ablation: {metric} relative to the full TGAT-BoostNet")
    PL.savefig(fig, f"03_ablation_{metric.lower().replace('-', '')}"); plt.show()

# %% [markdown]
# ## 11. Hyper-parameter sensitivity of the GBDT stage
# One parameter at a time around the defaults, GAT embeddings fixed (cached, 5 seeds). **Post-hoc analysis — not
# used to choose the reported configuration.** Two evaluation sets are shown: inner validation (fit on steps
# 1–29, score steps 30–34; uses only training data) and the test period.

# %%
from src.evaluation.tgat_boostnet import DEFAULT_GBDT, make_xgb
Xs = standardise(data)[0]
GRID = {"w_illicit": [1, 2, 4.32, 7.63, 15], "max_depth": [3, 4, 6, 8, 10], "n_estimators": [100, 200, 300, 600],
        "learning_rate": [0.02, 0.05, 0.1, 0.2]}
DEF = dict(DEFAULT_GBDT, w_illicit=4.32)
sens_path = os.path.join(B.RESULTS_DIR, "sensitivity_gbdt.csv")
if os.path.exists(sens_path):
    sens = pd.read_csv(sens_path)
else:
    rows = []
    in_tr_steps = data.time_steps[tr]
    for s in SEEDS:
        e_tr, e_te = B.load_embeddings(OURS, s)
        F_tr, F_te = np.hstack([Xs[tr], e_tr]), np.hstack([Xs[te], e_te])
        fit_m, val_m = in_tr_steps <= 29, in_tr_steps >= 30
        for prm, vals in GRID.items():
            for v in vals:
                params = dict(DEF); params[prm] = v
                w = params.pop("w_illicit")
                for where, (Xa, ya, Xb, yb) in {"val 30-34": (F_tr[fit_m], data.y[tr][fit_m], F_tr[val_m], data.y[tr][val_m]),
                                                "test 35-49": (F_tr, data.y[tr], F_te, y_te)}.items():
                    clf = make_xgb(params, w, s, DEVICE); clf.fit(Xa, ya)
                    mm = full_metrics(yb, clf.predict_proba(Xb)[:, 1])
                    rows.append(dict(seed=s, param=prm, value=v, eval=where, f1=mm["f1"], pr_auc=mm["pr_auc"]))
    sens = pd.DataFrame(rows)
    if not QUICK:
        sens.to_csv(sens_path, index=False)
for where in ("val 30-34", "test 35-49"):
    fig = PL.sensitivity_grid(sens[sens["eval"] == where], list(GRID), DEF, {"f1": "F1", "pr_auc": "PR-AUC"})
    fig.suptitle(f"GBDT sensitivity, evaluated on {where} (5 seeds, band = ±1 sd)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(); PL.savefig(fig, f"03_sensitivity_{where.split()[0]}"); plt.show()
sens_tab = sens.groupby(["eval", "param", "value"])[["f1", "pr_auc"]].agg(["mean", "std"])
sens_tab.columns = [f"{a}_{b}" for a, b in sens_tab.columns]
PL.save_table(sens_tab, "03_sensitivity", "GBDT-stage sensitivity (post-hoc)", png=False)

# %% [markdown]
# ## 12. Efficiency: latency / throughput vs F1 (Pareto)
# Latency = median time to score one complete test time-step snapshot. Bubble area ∝ log model size.
# Reference timings come from the idle-machine re-run of seed 42 (`results/timing/`) when available;
# otherwise from the seed-42 training run (which shared the CPU with other workers).

# %%
def timing(k):
    p = os.path.join(B.RESULTS_DIR, "timing", "runs", k, "seed_42.json")
    return (json.load(open(p)), "idle re-run") if os.path.exists(p) else (runs[k][SEEDS[0]], "training run")
eff = {}
for k in keys:
    t, srcname = timing(k)
    eff[LAB[k]] = {"F1": MF[k].f1.mean(), "PR-AUC": MF[k].pr_auc.mean(), "latency_ms": t["latency_ms_median"],
                   "latency_p95_ms": t["latency_ms_p95"], "throughput_tx_s": t["throughput_tx_per_s"],
                   "train_s": t["train_seconds"], "params": t["n_params"], "params_unit": t.get("n_params_note"),
                   "size_kb": t["size_bytes"] / 1024, "timing_source": srcname}
eff = pd.DataFrame(eff).T
for c in ("F1", "PR-AUC", "latency_ms", "latency_p95_ms", "throughput_tx_s", "train_s", "params", "size_kb"):
    eff[c] = eff[c].astype(float)
eff = eff.sort_values("F1", ascending=False)
PL.save_table(eff, "03_efficiency", "Efficiency (device: %s)" % runs[OURS][SEEDS[0]].get("device"),
              highlight_index=LAB[OURS], float_fmt="{:.3f}")
display(eff)
fig = PL.pareto(eff.assign(size_bytes=eff.size_kb * 1024), "latency_ms", "F1", "size_bytes", LAB[OURS],
                "median latency per time-step snapshot (ms, log)", "test illicit F1")
PL.savefig(fig, "03_pareto_latency_f1"); plt.show()
fig = PL.pareto(eff.assign(size_bytes=eff.size_kb * 1024, neg_tp=1.0 / eff.throughput_tx_s), "neg_tp", "F1", "size_bytes",
                LAB[OURS], "seconds per transaction = 1 / throughput (log)", "test illicit F1")
PL.savefig(fig, "03_pareto_throughput_f1"); plt.show()

# %% [markdown]
# ## 13. Learning curves of the strongest models
# Test F1 when trained on labelled steps 1..k only (5 seeds; ±1 sd). TGAT-BoostNet keeps its cached GAT
# embeddings (GAT trained on 1–34) and refits only the GBDT, so its curve isolates the booster's data needs.

# %%
lc_path = os.path.join(B.RESULTS_DIR, "learning_curves.csv")
LC_MODELS = [OURS] + [k for k in featured if k in ("rf", "xgb", "lgbm", "gbdt165_w", "catboost")][:3]
if os.path.exists(lc_path):
    lcd = pd.read_csv(lc_path)
else:
    rows = []
    for kstep in (10, 15, 20, 25, 30, 34):
        m_tr = data.time_steps[tr] <= kstep
        for s in SEEDS:
            for k in LC_MODELS:
                if k == OURS:
                    e_tr, e_te = B.load_embeddings(OURS, s)
                    clf = make_xgb(DEFAULT_GBDT, 4.32, s, DEVICE)
                    clf.fit(np.hstack([Xs[tr], e_tr])[m_tr], data.y[tr][m_tr])
                    p = clf.predict_proba(np.hstack([Xs[te], e_te]))[:, 1]
                else:
                    clf = B._tabular(k, s, DEVICE); clf.fit(Xs[tr][m_tr], data.y[tr][m_tr])
                    p = clf.predict_proba(Xs[te])[:, 1]
                rows.append(dict(model=k, seed=s, last_step=kstep, f1=full_metrics(y_te, p)["f1"]))
    lcd = pd.DataFrame(rows)
    if not QUICK:
        lcd.to_csv(lc_path, index=False)
fig, ax = plt.subplots(figsize=(6.5, 4))
cmap = PL.colors_for(LC_MODELS, OURS, [k for k in LC_MODELS if k != OURS])
for i, k in enumerate(LC_MODELS):
    g = lcd[lcd.model == k].groupby("last_step").f1
    ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), color=cmap[k], lw=2.2 if k == OURS else 1.2, marker="o", ms=3,
                capsize=2, ls="-" if k == OURS else PL.LINESTYLES[1 + i % 4], label=LAB[k])
ax.set(xlabel="trained on labelled steps 1..k", ylabel="test illicit F1", title="Learning curves (5 seeds)")
ax.legend(); fig.tight_layout(); PL.savefig(fig, "03_learning_curves"); plt.show()

# %% [markdown]
# ## 14. Results summary (generated from the numbers above)

# %%
def rank_of(metric):
    col = mean[metric]
    asc = METRIC_DIRECTION[metric] < 0
    order_ = col.sort_values(ascending=asc)
    return int(list(order_.index).index(LAB[OURS]) + 1), order_.index[0], order_.iloc[0]
n_models = len(mean)
lines = ["| Metric | TGAT-BoostNet (mean ± sd) | Rank (of %d reproduced) | Best reproduced model | Best value |" % n_models,
         "|---|---|---|---|---|"]
wins, losses = [], []
for c in SHOW:
    r, best, bv = rank_of(c)
    lines.append(f"| {METRIC_LABELS[c]} | {mean.loc[LAB[OURS], c]:.3f} ± {std.loc[LAB[OURS], c]:.3f} | {r} | {best} | {bv:.3f} |")
    (wins if r == 1 else losses).append((METRIC_LABELS[c], r, best))
table_md = "\n".join(lines)
sig_better = list(sig.index[(sig["p (F1 boot) Holm"] < 0.05) & (sig["ΔF1"] > 0)])
sig_worse = list(sig.index[(sig["p (F1 boot) Holm"] < 0.05) & (sig["ΔF1"] < 0)])
not_sig = list(sig.index[sig["p (F1 boot) Holm"] >= 0.05])
lit_best = lit.loc[lit.f1_illicit.idxmax()]
f1_ours = mean.loc[LAB[OURS], "f1"]
para = (f"Among {n_models} reproduced models on the identical split, TGAT-BoostNet ranks "
        f"**#{rank_of('f1')[0]} on illicit F1** ({f1_ours:.3f}), #{rank_of('pr_auc')[0]} on PR-AUC and "
        f"#{rank_of('roc_auc')[0]} on ROC-AUC. "
        + (f"It is first on: {', '.join(w[0] for w in wins)}. " if wins else "It is not first on any reported metric. ")
        + (f"Where it is not first: " + "; ".join(f"{m} (rank {r}, best = {b})" for m, r, b in losses) + ". " if losses else "")
        + f"Its F1 is significantly higher (paired bootstrap, Holm-corrected p < 0.05) than {len(sig_better)} baselines"
        + (f" ({', '.join(sig_better)})" if sig_better else "")
        + f", not significantly different from {len(not_sig)}"
        + (f" ({', '.join(not_sig)})" if not_sig else "")
        + f", and significantly lower than {len(sig_worse)}" + (f" ({', '.join(sig_worse)})" if sig_worse else "") + ". "
        + f"The highest published F1 on this split in our literature table is {lit_best.f1_illicit:.3f} "
        f"({lit_best.model}, {lit_best.features}, [{lit_best.source}], reported in paper, not reproduced here); "
        + ("TGAT-BoostNet's reproduced F1 is above it. " if f1_ours > lit_best.f1_illicit else
           "TGAT-BoostNet's reproduced F1 is below it. ")
        + "Like every model here, it fails after the step-43 dark-market shutdown (see §6).")
display(Markdown("### Where TGAT-BoostNet ranks\n\n" + table_md + "\n\n" + para))
with open(os.path.join(PL.TAB_DIR, "03_results_summary.md"), "w", encoding="utf-8") as f:
    f.write("# Results summary (auto-generated by notebooks/03_model_comparison.ipynb)\n\n" + table_md + "\n\n" + para + "\n")

# %% [markdown]
# ## 15. Download figures and tables

# %%
import zipfile
zp = os.path.join(ROOT, "03_model_comparison_outputs.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in (PL.FIG_DIR, PL.TAB_DIR):
        for f in sorted(os.listdir(d)):
            if f.startswith("03_"):
                zf.write(os.path.join(d, f), os.path.join(os.path.basename(d), f))
print("wrote", zp)
if IN_COLAB:
    from google.colab import files
    files.download(zp)
