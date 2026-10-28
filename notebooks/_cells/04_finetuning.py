# %% [markdown]
# # 04 · Fine-tuning TGAT-BoostNet — fairly
# **Course 21CSC314P (SRMIST) · Distributed Streaming Anomaly Detection on Blockchain Transaction Graphs**
#
# Notebooks 02/03 use settings fixed a priori. Here TGAT-BoostNet is **fine-tuned**, and so are the three
# strongest baselines, with the **same budget**. Otherwise a "win" would only mean "we tuned and they didn't".
#
# | | |
# |---|---|
# | Search split | fit = labelled steps 1–29, select = labelled steps 30–34 (inside the training period) |
# | Test | steps 35–49, scored **once**, after selection, at the threshold chosen on validation |
# | Budget | 40 random configurations per family (same seeded space for TGAT-BoostNet's GBDT and XGBoost), ranked by validation PR-AUC; threshold = validation-F1 maximiser |
# | TGAT-BoostNet extra stage | GAT grid: heads {4, 8} × epochs {100, 200} (cross-fitted embeddings inside 1–29) |
# | Final | refit on labelled 1–34, 5 seeds (42, 123, 456, 789, 2024) |
# | Runtime | **use a GPU runtime** (Runtime → Change runtime type → T4): ~45–60 min from scratch; a few minutes with cached `results/tuning/` |
#
# Caveat recorded up front: validation steps 30–34 are easy (F1 ≈ 0.97 for strong models), so validation
# separates candidates only weakly, and thresholds chosen there may not transfer past the step-43 shift.

# %% INCLUDE_SETUP

# %% [markdown]
# ## 1. Data

# %%
from src.evaluation.data import load_elliptic, SEEDS
from src.evaluation import baselines as B, stats as ST, tuning as TU
from src.evaluation.metrics import full_metrics, METRIC_LABELS, fast_counts_metrics, per_timestep
data = load_elliptic()
y_te, steps_te = data.y[data.test_mask], data.time_steps[data.test_mask]
fit, val = data.mask(TU.FIT_STEPS), data.mask(TU.VAL_STEPS)
if QUICK:  # smoke test only: tiny search, 3-epoch GAT
    TU.N_TRIALS, TU.GAT_GRID = 4, [dict(heads=4, epochs=3)]
print(f"search fit rows {fit.sum()} ({(data.y[fit] == 1).sum()} illicit) | validation rows {val.sum()} "
      f"({(data.y[val] == 1).sum()} illicit) | test rows {data.test_mask.sum()} (not used until §4)")
print("trials per family:", TU.N_TRIALS, "| device:", DEVICE)

# %% [markdown]
# ## 2. Search (validation only)

# %%
t0 = time.time()
sel = TU.tune_all(data, DEVICE)
print(f"search done in {time.time() - t0:.0f}s (0 s = loaded from results/tuning/)")
display(pd.read_csv(os.path.join(TU.TUNE_DIR(), "gat_stage.csv")))
sel_tab = pd.DataFrame({B.MODELS[f][0] if f in B.MODELS else f: {
    "validation PR-AUC": s["val_pr_auc"], "validation F1 @ threshold": s["val_f1"], "threshold": s["threshold"],
    "selected params": json.dumps({**s["params"], **s.get("gat", {})})} for f, s in sel.items()}).T
PL.save_table(sel_tab, "04_selected_configs", "Configurations selected on validation steps 30-34", float_fmt="{:.4f}", png=False)
display(sel_tab)

# %%
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
for ax, fam in zip(axes, ("tgat_boostnet", "xgb", "lgbm", "rf")):
    s = pd.read_csv(os.path.join(TU.TUNE_DIR(), f"search_{fam}.csv")).sort_values("trial")
    best = s.val_pr_auc.cummax()
    ax.scatter(s.trial, s.val_pr_auc, s=10, color=PL.OURS_COLOR if fam == "tgat_boostnet" else PL.CATEGORICAL[0], alpha=0.6)
    ax.plot(s.trial, best, color=PL.INK2, lw=1, label="best so far")
    ax.set(title=B.MODELS[fam][0], xlabel="trial")
axes[0].set_ylabel("validation PR-AUC"); axes[0].legend(loc="lower right")
fig.suptitle("Random search on validation steps 30-34 (same 40-trial budget per family)", x=0.01, ha="left", fontsize=9)
fig.tight_layout(); PL.savefig(fig, "04_search_traces"); plt.show()

# %% [markdown]
# ## 3. Final refits on steps 1–34 (5 seeds)

# %%
tuned = TU.final_runs(sel, data, DEVICE)
untuned_keys = ["tgat_boostnet", "xgb", "lgbm", "rf"]
untuned = {k: {s: B.run(k, data, s, DEVICE) for s in B.seeds_for(k)} for k in untuned_keys}

# %% [markdown]
# ## 4. Test results: untuned vs tuned
# Untuned models use threshold 0.5; tuned models use their validation threshold (shown).

# %%
def frame(runs_, thr):
    return pd.DataFrame([full_metrics(y_te, r["p_test"], thr) for r in runs_.values()])
rows, MF = {}, {}
for k in untuned_keys:
    MF[f"{k} (untuned)"] = frame(untuned[k], 0.5)
for k, rr in tuned.items():
    fam = k.replace("_tuned", "")
    MF[f"{fam} (tuned)"] = frame(rr, sel[fam]["threshold"])
SHOW = ["f1", "precision", "recall", "mcc", "roc_auc", "pr_auc", "balanced_accuracy"]
name = lambda lab: lab.replace(lab.split(" ")[0], B.MODELS[lab.split(" ")[0]][0])
res = pd.DataFrame({name(k): {METRIC_LABELS[c]: f"{v[c].mean():.3f} ± {v[c].std(ddof=1):.3f}" for c in SHOW} for k, v in MF.items()}).T
res_num = pd.DataFrame({name(k): v[SHOW].mean() for k, v in MF.items()}).T.sort_values("f1", ascending=False)
res = res.loc[res_num.index]
res.insert(0, "threshold", [0.5 if "untuned" in i else sel[next(f for f in sel if B.MODELS[f][0] in i)]["threshold"] for i in res.index])
OURS_T = name("tgat_boostnet (tuned)")
PL.save_table(res, "04_tuned_vs_untuned", "Test steps 35-49: untuned (threshold 0.5) vs validation-tuned (5 seeds)",
              highlight_index=[OURS_T, name("tgat_boostnet (untuned)")])
display(res)

# %%
d = res_num.copy()
fig, ax = plt.subplots(figsize=(8, 3.6))
fams = ["tgat_boostnet", "lgbm", "rf", "xgb"]
for i, fam in enumerate(fams):
    a, b = d.loc[name(f"{fam} (untuned)"), "f1"], d.loc[name(f"{fam} (tuned)"), "f1"]
    col = PL.OURS_COLOR if fam == "tgat_boostnet" else PL.CATEGORICAL[i - 1]
    ax.annotate("", (b, i), (a, i), arrowprops=dict(arrowstyle="->", color=col, lw=2))
    ax.scatter([a], [i], color="white", edgecolor=col, s=60, zorder=3, lw=1.5)
    ax.scatter([b], [i], color=col, s=60, zorder=3)
    ax.text(max(a, b) + 0.003, i, f"{a:.3f} → {b:.3f}", va="center", fontsize=7.5, color=PL.INK2)
ax.set_yticks(range(len(fams))); ax.set_yticklabels([B.MODELS[f][0] for f in fams])
ax.set_xlabel("test illicit F1 (mean of 5 seeds); hollow = untuned, filled = tuned")
ax.set_title("Effect of validation-only tuning", loc="left"); fig.tight_layout()
PL.savefig(fig, "04_tuning_effect"); plt.show()

# %% [markdown]
# ## 5. Significance: tuned TGAT-BoostNet vs each tuned baseline
# Paired bootstrap (2,000 resamples) of the 5-seed-mean F1 / PR-AUC difference; exact McNemar on the seed-averaged
# predictions; Holm-corrected.

# %%
from sklearn.metrics import average_precision_score
thr_of = {f"{f}_tuned": sel[f]["threshold"] for f in sel}
P = {k: [r["p_test"] for r in v.values()] for k, v in tuned.items()}
ours = "tgat_boostnet_tuned"
sig = {}
for k in P:
    if k == ours:
        continue
    f1a = lambda y, p, t=thr_of[ours]: fast_counts_metrics(y, (p >= t).astype(int))["f1"]
    f1b = lambda y, p, t=thr_of[k]: fast_counts_metrics(y, (p >= t).astype(int))["f1"]
    rng = np.random.default_rng(0); n = len(y_te); diffs = []
    for _ in range(300 if QUICK else 2000):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean([f1a(y_te[idx], p[idx]) for p in P[ours]]) - np.mean([f1b(y_te[idx], p[idx]) for p in P[k]]))
    diffs = np.array(diffs)
    obs = np.mean([f1a(y_te, p) for p in P[ours]]) - np.mean([f1b(y_te, p) for p in P[k]])
    ap = ST.paired_bootstrap(y_te, P[ours], P[k], lambda y, p: average_precision_score(y, p), n_boot=300 if QUICK else 500)
    mc = ST.mcnemar(y_te, (np.mean(P[ours], 0) >= thr_of[ours]).astype(int), (np.mean(P[k], 0) >= thr_of[k]).astype(int))
    sig[B.MODELS[k.replace("_tuned", "")][0] + " (tuned)"] = {
        "ΔF1": obs, "ΔF1 95% CI": f"[{np.quantile(diffs, .025):+.3f}, {np.quantile(diffs, .975):+.3f}]",
        "p F1": float(min(1, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
        "ΔPR-AUC": ap["diff"], "ΔPR-AUC 95% CI": f"[{ap['ci_low']:+.3f}, {ap['ci_high']:+.3f}]", "p PR-AUC": ap["p_value"],
        "p McNemar": mc["p_value"]}
sig = pd.DataFrame(sig).T
for c in ("p F1", "p PR-AUC", "p McNemar"):
    sig[c + " (Holm)"] = sig.index.map(ST.holm(sig[c].astype(float).to_dict()))
PL.save_table(sig, "04_significance_tuned", "Tuned TGAT-BoostNet vs tuned baselines (positive = ours better)", float_fmt="{:.4f}")
display(sig)

# %% [markdown]
# ## 6. Per-time-step F1 after tuning

# %%
rows = []
for k, rr in tuned.items():
    for s, r in rr.items():
        rows += [dict(x, model=k, seed=s) for x in per_timestep(y_te, r["p_test"], steps_te, thr_of[k])]
pts = pd.DataFrame(rows)
labels = {k: B.MODELS[k.replace("_tuned", "")][0] + " (tuned)" for k in tuned}
counts = pts[(pts.model == ours) & (pts.seed == SEEDS[0])].set_index("time_step").n_illicit
fig = PL.per_timestep_lines(pts, ours, [k for k in tuned if k != ours], labels, counts=counts)
PL.savefig(fig, "04_per_timestep_tuned"); plt.show()

# %% [markdown]
# ## 7. Summary (generated)

# %%
f1 = res_num["f1"]
rank = int(list(f1.index).index(OURS_T) + 1)
gain = f1[OURS_T] - f1[name("tgat_boostnet (untuned)")]
best_lab = f1.index[0]
better = [i for i, r in sig.iterrows() if float(r["p F1 (Holm)"]) < 0.05 and r["ΔF1"] > 0]
worse = [i for i, r in sig.iterrows() if float(r["p F1 (Holm)"]) < 0.05 and r["ΔF1"] < 0]
txt = (f"Validation-only tuning changed TGAT-BoostNet's test F1 from {f1[name('tgat_boostnet (untuned)')]:.3f} to "
       f"{f1[OURS_T]:.3f} ({gain:+.3f}). Among the 8 untuned/tuned variants of the four strongest families it ranks "
       f"**#{rank} on F1** (best: {best_lab}, {f1.iloc[0]:.3f}); PR-AUC {res_num.loc[OURS_T, 'pr_auc']:.3f}, ROC-AUC "
       f"{res_num.loc[OURS_T, 'roc_auc']:.3f}. Against the equally tuned baselines its F1 is significantly higher than "
       f"{len(better)} ({', '.join(better) or 'none'}), significantly lower than {len(worse)} ({', '.join(worse) or 'none'}), "
       f"and not significantly different from the rest (paired bootstrap, Holm-corrected).")
display(Markdown(txt))
open(os.path.join(PL.TAB_DIR, "04_summary.md"), "w", encoding="utf-8").write(txt + "\n")

# %% [markdown]
# ## 8. Download

# %%
import zipfile
zp = os.path.join(ROOT, "04_finetuning_outputs.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    for d_ in (PL.FIG_DIR, PL.TAB_DIR):
        for f in sorted(os.listdir(d_)):
            if f.startswith("04_"):
                zf.write(os.path.join(d_, f), os.path.join(os.path.basename(d_), f))
    for root_, _, files_ in os.walk(os.path.join(B.RESULTS_DIR, "tuning")):
        for f in files_:
            zf.write(os.path.join(root_, f), os.path.relpath(os.path.join(root_, f), ROOT))
    for k in tuned:
        for root_, _, files_ in os.walk(os.path.join(B.RESULTS_DIR, "runs", k)):
            for f in files_:
                zf.write(os.path.join(root_, f), os.path.relpath(os.path.join(root_, f), ROOT))
print("wrote", zp, "- contains figures, tables and the tuned runs (copy results/ back into the repo to cache them)")
if IN_COLAB:
    from google.colab import files
    files.download(zp)
