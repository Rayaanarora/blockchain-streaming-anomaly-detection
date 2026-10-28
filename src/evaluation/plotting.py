"""
Publication figures and tables for notebooks 01-03.

Style: 300 dpi, PNG + PDF for every figure; one recessive grid; categorical hues from a palette that
passes colour-vision-deficiency checks (validated with the dataviz skill's validator: worst adjacent
CVD dE 17.6, normal-vision dE 24.0). TGAT-BoostNet always wears the same orange and is drawn on
top; up to five "featured" baselines get the next hues in fixed order; every other model is grey.
Identity is never colour-only: legends are always present and line styles differ.
"""
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.data import REPO_ROOT

FIG_DIR = os.environ.get("BSAD_FIG_DIR", os.path.join(REPO_ROOT, "figures"))
TAB_DIR = os.environ.get("BSAD_TAB_DIR", os.path.join(REPO_ROOT, "tables"))

OURS_COLOR = "#eb6834"
CATEGORICAL = ["#2a78d6", "#1baf7a", "#6250d6", "#e87ba4", "#008300"]
GREY = "#9a9994"
INK, INK2 = "#0b0b0b", "#52514e"
SEQ_CMAP = "Blues"
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
        "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
        "legend.fontsize": 7.5, "lines.linewidth": 1.4, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def colors_for(keys: Sequence[str], ours: str, featured: Sequence[str] = ()) -> Dict[str, str]:
    cmap = {}
    feat = [k for k in featured if k != ours][:len(CATEGORICAL)]
    for k in keys:
        if k == ours:
            cmap[k] = OURS_COLOR
        elif k in feat:
            cmap[k] = CATEGORICAL[feat.index(k)]
        else:
            cmap[k] = GREY
    return cmap


def savefig(fig, name: str, outdir: Optional[str] = None):
    outdir = outdir or FIG_DIR
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    return os.path.join(outdir, f"{name}.png")


def save_table(df: pd.DataFrame, name: str, caption: str = "", highlight_index=None, outdir: Optional[str] = None,
               float_fmt: str = "{:.3f}", png: bool = True):
    """Write CSV, LaTeX, styled HTML and a rendered PNG of ``df``; returns the Styler for display."""
    import matplotlib.pyplot as plt
    outdir = outdir or TAB_DIR
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"{name}.csv"))
    fmt_df = df.copy()
    for c in fmt_df.columns:
        if pd.api.types.is_float_dtype(fmt_df[c]):
            fmt_df[c] = fmt_df[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    with open(os.path.join(outdir, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write(fmt_df.to_latex(escape=True, caption=caption or None, label=f"tab:{name}"))
    sty = fmt_df.style.set_caption(caption).set_table_styles([
        {"selector": "caption", "props": "caption-side: top; font-weight: 600; text-align: left;"},
        {"selector": "th", "props": "background:#f2f1ee; padding:4px 8px; text-align:left;"},
        {"selector": "td", "props": "padding:4px 8px; font-variant-numeric: tabular-nums;"},
        {"selector": "", "props": "border-collapse: collapse; font-family: sans-serif; font-size: 12px;"}])
    if highlight_index is not None:
        hl = set([highlight_index] if not isinstance(highlight_index, (list, tuple, set)) else highlight_index)
        sty = sty.apply(lambda r: ["background:#fde6dc; font-weight:600" if r.name in hl else "" for _ in r],
                        axis=1)
    with open(os.path.join(outdir, f"{name}.html"), "w", encoding="utf-8") as f:
        f.write(sty.to_html())
    if png:
        n_rows, n_cols = fmt_df.shape
        fig, ax = plt.subplots(figsize=(min(1.2 + 1.05 * (n_cols + 1), 22), 0.32 * (n_rows + 2) + 0.4))
        ax.axis("off")
        cell = fmt_df.astype(str).to_numpy()
        tbl = ax.table(cellText=cell, rowLabels=[str(i) for i in fmt_df.index], colLabels=list(fmt_df.columns),
                       loc="center", cellLoc="right", rowLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.auto_set_column_width(list(range(-1, n_cols)))
        for (r, c), cl in tbl.get_celld().items():
            cl.set_edgecolor("#e4e3df")
            if r == 0:
                cl.set_facecolor("#f2f1ee")
                cl.set_text_props(weight="bold")
            if highlight_index is not None and r > 0 and fmt_df.index[r - 1] in hl:
                cl.set_facecolor("#fde6dc")
                cl.set_text_props(weight="bold")
        if caption:
            ax.set_title(caption, loc="left", fontsize=9, color=INK)
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    return sty


def _mean_band(xgrid, curves):
    arr = np.vstack(curves)
    m = arr.mean(0)
    if len(curves) < 2:
        return m, None, None
    from scipy import stats
    half = stats.t.ppf(0.975, len(curves) - 1) * arr.std(0, ddof=1) / np.sqrt(len(curves))
    return m, np.clip(m - half, 0, 1), np.clip(m + half, 0, 1)


def roc_pr_overlay(y, probs: Dict[str, List[np.ndarray]], labels: Dict[str, str], ours: str,
                   featured: Sequence[str] = ()):
    """Mean ROC and PR curves over seeds; shaded 95 % CI of the seed mean for ours + featured models."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    cmap = colors_for(list(probs), ours, featured)
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    order = [k for k in probs if cmap[k] == GREY] + [k for k in featured if k in probs and k != ours] + [ours]
    for k in order:
        tprs, precs, aucs, aps = [], [], [], []
        for p in probs[k]:
            fpr, tpr, _ = roc_curve(y, p)
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(auc(fpr, tpr))
            pr, rc, _ = precision_recall_curve(y, p)
            precs.append(np.interp(grid, rc[::-1], pr[::-1]))
            aps.append(auc(rc, pr))
        hi = cmap[k] != GREY
        lw, z, alpha = (2.4, 5, 1) if k == ours else ((1.5, 4, 1) if hi else (0.8, 2, 0.55))
        ls = "-" if k == ours or not hi else LINESTYLES[1 + list(featured).index(k) % 4]
        for ax, curves, val, nm in ((axes[0], tprs, np.mean(aucs), "AUC"), (axes[1], precs, np.mean(aps), "AP")):
            m, lo, up = _mean_band(grid, curves)
            ax.plot(grid, m, color=cmap[k], lw=lw, ls=ls, zorder=z, alpha=alpha,
                    label=f"{labels[k]} ({nm} {val:.3f})" if hi else None)
            if hi and lo is not None:
                ax.fill_between(grid, lo, up, color=cmap[k], alpha=0.15, lw=0, zorder=z - 1)
    axes[0].plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=":")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC, test steps 35-49")
    axes[1].axhline(np.mean(y), color=INK2, lw=0.6, ls=":")
    axes[1].set(xlabel="Recall (illicit)", ylabel="Precision (illicit)", title="Precision-recall, test steps 35-49")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
    axes[0].plot([], [], color=GREY, lw=0.8, label="other baselines")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower left")
    fig.text(0.01, -0.02, "Lines: mean over seeds; bands: 95 % CI of the seed mean (t-interval).", fontsize=7,
             color=INK2)
    fig.tight_layout()
    return fig


def radar(df: pd.DataFrame, ours: str, featured: Sequence[str], labels: Dict[str, str]):
    """df: rows = models, columns = metrics (already oriented so higher is better, scaled 0-1)."""
    import matplotlib.pyplot as plt
    keys = [k for k in list(featured) + [ours] if k in df.index]
    cmap = colors_for(keys, ours, featured)
    cats = list(df.columns)
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(5.6, 5.6))
    ax = fig.add_subplot(111, polar=True)
    for i, k in enumerate(keys):
        v = df.loc[k].tolist()
        v += v[:1]
        ax.plot(ang, v, color=cmap[k], lw=2.4 if k == ours else 1.3, ls="-" if k == ours else LINESTYLES[1 + i % 4],
                label=labels[k], zorder=5 if k == ours else 3)
        if k == ours:
            ax.fill(ang, v, color=cmap[k], alpha=0.12)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(cats, fontsize=7.5)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    return fig


def heatmap_leaderboard(values: pd.DataFrame, ranks: pd.DataFrame, ours_label: str, title: str):
    """Annotated models x metrics heatmap; colour = rank (1 = best, darkest), text = value."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.0 + 0.78 * values.shape[1], 0.9 + 0.36 * values.shape[0]))
    r = ranks.to_numpy(dtype=float)
    im = ax.imshow(-r, cmap=SEQ_CMAP, aspect="auto", vmin=-r.max(), vmax=-1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values.iat[i, j]
            dark = r[i, j] <= max(2, r.max() * 0.3)
            ax.text(j, i, f"{v:.3f}\n#{int(r[i, j])}", ha="center", va="center", fontsize=6.3,
                    color="white" if dark else INK)
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(values.columns, rotation=35, ha="right")
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(values.index)
    for t in ax.get_yticklabels():
        if t.get_text() == ours_label:
            t.set_color(OURS_COLOR)
            t.set_fontweight("bold")
    ax.grid(False)
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_ticks([-1, -r.max()])
    cb.set_ticklabels(["rank 1 (best)", f"rank {int(r.max())}"])
    fig.tight_layout()
    return fig


def per_timestep_lines(df: pd.DataFrame, ours: str, featured: Sequence[str], labels: Dict[str, str],
                       metric: str = "f1", counts: Optional[pd.Series] = None):
    """df columns: model, seed, time_step, <metric>. Lines = seed mean; bands = min-max over seeds."""
    import matplotlib.pyplot as plt
    keys = [k for k in featured if k in df.model.unique()] + [ours]
    cmap = colors_for(keys, ours, featured)
    nrows = 2 if counts is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 4.6 if nrows == 2 else 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]} if nrows == 2 else None)
    ax = axes[0] if nrows == 2 else axes
    for i, k in enumerate(keys):
        g = df[df.model == k].groupby("time_step")[metric]
        m, lo, hi = g.mean(), g.min(), g.max()
        ax.plot(m.index, m.values, color=cmap[k], lw=2.4 if k == ours else 1.3, marker="o", ms=3.5,
                ls="-" if k == ours else LINESTYLES[1 + i % 4], label=labels[k], zorder=5 if k == ours else 3)
        ax.fill_between(m.index, lo.values, hi.values, color=cmap[k], alpha=0.15, lw=0)
    ax.axvline(43, color=INK2, lw=0.8, ls="--")
    ax.text(43.15, 0.97, "dark-market shutdown (t = 43)", fontsize=7, color=INK2, va="top")
    ax.set_ylabel(f"Illicit {metric.upper() if metric == 'f1' else metric}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", ncol=2)
    ax.set_title("Per-time-step performance on the test period (band: min-max over seeds)", loc="left")
    if nrows == 2:
        axes[1].bar(counts.index, counts.values, color=GREY, width=0.7)
        axes[1].set_ylabel("# illicit")
        axes[1].axvline(43, color=INK2, lw=0.8, ls="--")
        axes[1].set_xlabel("Time step")
    else:
        ax.set_xlabel("Time step")
    fig.tight_layout()
    return fig


def raincloud(df: pd.DataFrame, metric: str, order: List[str], ours_label: str, xlabel: str):
    """Half-violin + box + jittered seeds per model (df columns: label, seed, <metric>)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 1.2))
    rng = np.random.default_rng(0)
    for i, lab in enumerate(order):
        v = df.loc[df.label == lab, metric].to_numpy()
        col = OURS_COLOR if lab == ours_label else CATEGORICAL[0]
        if len(v) > 1 and np.ptp(v) > 0:
            parts = ax.violinplot(v, positions=[i], vert=False, showextrema=False, widths=0.8)
            for b in parts["bodies"]:
                verts = b.get_paths()[0].vertices
                verts[:, 1] = np.clip(verts[:, 1], i, None)
                b.set_facecolor(col)
                b.set_alpha(0.3)
                b.set_edgecolor("none")
        ax.boxplot(v, positions=[i - 0.15], vert=False, widths=0.12, showfliers=False,
                   medianprops=dict(color=INK), boxprops=dict(color=INK2), whiskerprops=dict(color=INK2),
                   capprops=dict(color=INK2))
        ax.scatter(v, i - 0.32 + rng.uniform(-0.05, 0.05, len(v)), s=12, color=col, zorder=4,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for t in ax.get_yticklabels():
        if t.get_text() == ours_label:
            t.set_color(OURS_COLOR)
            t.set_fontweight("bold")
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.7, len(order) - 0.2)
    fig.tight_layout()
    return fig


def calibration_and_threshold(y, probs: Dict[str, np.ndarray], labels: Dict[str, str], ours: str,
                              featured: Sequence[str] = (), n_bins: int = 15):
    """Reliability diagram for several models + P/R/F1 vs threshold for ours."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    cmap = colors_for(list(probs), ours, featured)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=INK2, ls=":", lw=0.8, label="perfect calibration")
    for i, (k, p) in enumerate(probs.items()):
        fr, mp = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
        ax.plot(mp, fr, marker="o", ms=3, color=cmap[k], lw=2.2 if k == ours else 1.2,
                ls="-" if k == ours else LINESTYLES[1 + i % 4], label=labels[k], zorder=5 if k == ours else 3)
    ax.set(xlabel="Mean predicted P(illicit)", ylabel="Observed illicit fraction",
           title=f"Reliability ({n_bins} equal-width bins, as in ECE)")
    ax.legend(loc="upper left")
    ax = axes[1]
    p = probs[ours]
    th = np.linspace(0.01, 0.99, 99)
    P, R, F = [], [], []
    for t in th:
        pr = p >= t
        tp = (pr & (y == 1)).sum()
        P.append(tp / max(pr.sum(), 1))
        R.append(tp / max((y == 1).sum(), 1))
        F.append(0 if P[-1] + R[-1] == 0 else 2 * P[-1] * R[-1] / (P[-1] + R[-1]))
    ax.plot(th, P, color=CATEGORICAL[0], ls="--", label="precision")
    ax.plot(th, R, color=CATEGORICAL[1], ls="-.", label="recall")
    ax.plot(th, F, color=OURS_COLOR, lw=2.2, label="F1")
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    tb = th[int(np.argmax(F))]
    ax.annotate(f"F1 max {max(F):.3f} at t={tb:.2f}\n(post-hoc, not used)", (tb, max(F)), (0.55, 0.25),
                fontsize=7, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.6))
    ax.set(xlabel="Decision threshold", ylabel="Score", title=f"Threshold sweep: {labels[ours]}", ylim=(0, 1.02))
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def confusion_grid(cms: Dict[str, np.ndarray], labels: Dict[str, str], ours: str, ncols: int = 4):
    """Row-normalised confusion matrices (rows = true class)."""
    import matplotlib.pyplot as plt
    keys = list(cms)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, keys):
        cm = cms[k].astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["licit", "illicit"], fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["licit", "illicit"], fontsize=7)
        ax.grid(False)
        ax.set_title(labels[k], fontsize=8, color=OURS_COLOR if k == ours else INK,
                     fontweight="bold" if k == ours else "normal")
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.supxlabel("Predicted", fontsize=8)
    fig.supylabel("True", fontsize=8)
    fig.tight_layout()
    return fig


def embedding_panels(panels: List[tuple], y: np.ndarray):
    """panels: [(title, Z (n,2))]; points coloured by class (licit blue, illicit orange)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.9))
    for ax, (title, Z) in zip(np.atleast_1d(axes), panels):
        for cls, col, lab, s in ((0, CATEGORICAL[0], "licit", 3), (1, OURS_COLOR, "illicit", 5)):
            m = y == cls
            ax.scatter(Z[m, 0], Z[m, 1], s=s, c=col, alpha=0.45 if cls == 0 else 0.8, lw=0, label=lab)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    np.atleast_1d(axes)[0].legend(loc="best", markerscale=3)
    fig.tight_layout()
    return fig


def shap_beeswarm(contrib: np.ndarray, X: np.ndarray, names: List[str], is_gat: np.ndarray, top: int = 20):
    """Beeswarm of per-row SHAP contributions (top features by mean |SHAP|) + GAT-vs-raw importance share."""
    import matplotlib.pyplot as plt
    imp = np.abs(contrib).mean(0)
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.28 * top + 1.4), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        v, x = contrib[:, j], X[:, j]
        lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
        c = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
        hist, edges = np.histogram(v, bins=50)
        b = np.clip(np.digitize(v, edges) - 1, 0, 49)
        jitter = rng.uniform(-1, 1, len(v)) * 0.38 * hist[b] / max(hist.max(), 1)
        ax.scatter(v, row + jitter, c=c, cmap="coolwarm", s=3, lw=0, alpha=0.7, rasterized=True)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[j] for j in order], fontsize=7)
    for t, j in zip(ax.get_yticklabels(), order):
        if is_gat[j]:
            t.set_color(OURS_COLOR)
            t.set_fontweight("bold")
    ax.set_xlabel("SHAP value (log-odds contribution to illicit)")
    ax.set_title("SHAP beeswarm, fused 197-d input (orange labels = GAT embedding dims)", loc="left")
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("feature value", fontsize=7)
    ax = axes[1]
    share = pd.Series({"Raw local f1-f93": imp[:93].sum(), "Raw aggregated f94-f165": imp[93:165].sum(),
                       "GAT embedding (32)": imp[is_gat].sum()})
    share = share / share.sum()
    cols = [CATEGORICAL[0], CATEGORICAL[1], OURS_COLOR]
    ax.barh(share.index, share.values, color=cols, height=0.55)
    for i, v in enumerate(share.values):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(share.values) * 1.3)
    ax.set_xlabel("Share of mean |SHAP|")
    ax.set_title("Importance by feature group", loc="left")
    fig.tight_layout()
    return fig


def lollipop(df: pd.DataFrame, value: str, err: Optional[str], ref: float, ref_label: str, xlabel: str,
             title: str):
    """Horizontal lollipop of ``value`` per row relative to a reference line (ablation chart)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(df) + 1.3))
    y = np.arange(len(df))
    for i, (lab, r) in enumerate(df.iterrows()):
        better = r[value] >= ref
        col = OURS_COLOR if r.get("is_ref", False) else (CATEGORICAL[1] if better else CATEGORICAL[0])
        ax.hlines(i, ref, r[value], color=col, lw=2)
        ax.plot(r[value], i, "o", color=col, ms=7, zorder=4, mec="white", mew=1.2)
        if err and not pd.isna(r[err]):
            ax.errorbar(r[value], i, xerr=r[err], color=INK2, lw=0.8, capsize=2, zorder=3)
        ax.text(r[value] + (0.004 if better else -0.004), i + 0.28, f"{r[value] - ref:+.3f}", fontsize=7,
                ha="left" if better else "right", color=INK2)
    ax.axvline(ref, color=OURS_COLOR, lw=1, ls="--")
    ax.text(ref, len(df) - 0.35, ref_label, color=OURS_COLOR, fontsize=7, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def pareto(df: pd.DataFrame, x: str, yname: str, size: str, ours_label: str, xlabel: str, ylabel: str,
           logx: bool = True):
    """Bubble chart (bubble area ~ model size) with the Pareto frontier (lower x, higher y is better)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    s = df[size].clip(lower=1)
    area = 30 + 900 * (np.log10(s) - np.log10(s).min()) / max(np.ptp(np.log10(s)), 1e-9)
    for lab, r in df.iterrows():
        ours = lab == ours_label
        ax.scatter(r[x], r[yname], s=area[lab], color=OURS_COLOR if ours else CATEGORICAL[0],
                   alpha=0.85 if ours else 0.45, edgecolor="white", lw=1, zorder=5 if ours else 3)
        ax.annotate(lab, (r[x], r[yname]), xytext=(5, 4), textcoords="offset points", fontsize=6.8,
                    color=OURS_COLOR if ours else INK, fontweight="bold" if ours else "normal")
    d = df.sort_values(x)
    front, best = [], -np.inf
    for lab, r in d.iterrows():
        if r[yname] > best:
            front.append((r[x], r[yname]))
            best = r[yname]
    fx, fy = zip(*front)
    ax.step(fx, fy, where="post", color=INK2, lw=0.9, ls="--", label="Pareto frontier")
    if logx:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    for v, lab in ((s.min(), "small"), (s.max(), "large")):
        ax.scatter([], [], s=30 + (900 if lab == "large" else 0), color=GREY, alpha=0.4,
                   label=f"{lab} model ({v / 1024:.0f} KB)")
    ax.legend(loc="lower right", labelspacing=1.6, borderpad=1)
    fig.tight_layout()
    return fig


def cd_diagram(avg_ranks: pd.Series, cd: float, ours_label: str, title: str):
    """Demsar-style critical-difference diagram: models on a rank axis, bars join groups within CD."""
    import matplotlib.pyplot as plt
    r = avg_ranks.sort_values()
    k = len(r)
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.23 * k + 1.2))
    ax.set_xlim(0.5, k + 0.5)
    ax.invert_xaxis()
    ax.set_ylim(-1.2 - 0.23 * k, 1.2)
    ax.axhline(0, color=INK, lw=1)
    for i in range(1, k + 1):
        ax.plot([i, i], [0, 0.12], color=INK, lw=0.8)
        ax.text(i, 0.25, str(i), ha="center", fontsize=7, color=INK2)
    half = int(np.ceil(k / 2))
    for j, (lab, v) in enumerate(r.items()):
        left = j < half
        yy = -0.35 - 0.23 * (j if left else k - 1 - j)
        xt = k + 0.4 if not left else 0.6
        ax.plot([v, v, xt], [0, yy, yy], color=OURS_COLOR if lab == ours_label else INK2, lw=0.8)
        ax.text(xt, yy, f" {lab} ({v:.2f}) " if not left else f" ({v:.2f}) {lab} ", va="center",
                ha="right" if not left else "left", fontsize=7,
                color=OURS_COLOR if lab == ours_label else INK, fontweight="bold" if lab == ours_label else "normal")
    # cliques: maximal runs of consecutive models whose rank span < CD
    vals = r.to_numpy()
    cliques, last_end = [], -1
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and j > last_end:
            cliques.append((i, j))
            last_end = j
    for n, (i, j) in enumerate(cliques):
        yy = -0.12 - 0.07 * n
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [yy, yy], color=INK, lw=2.5, solid_capstyle="round")
    ax.plot([1, 1 + cd], [0.85, 0.85], color=INK, lw=1.5)
    ax.text(1 + cd / 2, 0.95, f"CD = {cd:.2f}", ha="center", fontsize=7.5)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=9)
    fig.tight_layout()
    return fig


def sensitivity_grid(df: pd.DataFrame, params: List[str], defaults: Dict[str, float], metric_cols: Dict[str, str]):
    """Small multiples: one panel per hyper-parameter; lines = metrics (mean over seeds, band = +/-1 sd)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(params), figsize=(3.3 * len(params), 3.2), sharey=True)
    for ax, prm in zip(np.atleast_1d(axes), params):
        d = df[df.param == prm]
        for i, (col, lab) in enumerate(metric_cols.items()):
            g = d.groupby("value")[col]
            m, s = g.mean(), g.std().fillna(0)
            c = OURS_COLOR if i == 0 else CATEGORICAL[i - 1]
            ax.plot(m.index, m.values, marker="o", ms=3.5, color=c, ls=LINESTYLES[i % 5], label=lab)
            ax.fill_between(m.index, m - s, m + s, color=c, alpha=0.15, lw=0)
        ax.axvline(defaults[prm], color=INK2, lw=0.8, ls=":")
        ax.set_xlabel(prm + " (dotted = default)")
        if prm in ("learning_rate", "w_illicit", "n_estimators"):
            ax.set_xscale("log")
    np.atleast_1d(axes)[0].set_ylabel("score")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    fig.tight_layout()
    return fig
