"""
Significance tests and rank summaries for notebook 03.

* McNemar's exact test on paired hard predictions (same test transactions).
* Paired bootstrap of a metric difference, resampling test transactions; each replicate averages
  the metric over the 5 seeds of each model, matching how the tables report means.
* Holm-Bonferroni correction across the baselines compared with TGAT-BoostNet.
* Friedman test + Nemenyi critical difference (Demsar, JMLR 2006) with the 15 test time steps as
  blocks, i.e. "does one model rank consistently above another across time?".
"""
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact (binomial) McNemar test; b = A right & B wrong, c = A wrong & B right."""
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    ra, rb = pred_a == y, pred_b == y
    b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    chi2 = 0.0 if b + c == 0 else (abs(b - c) - 1) ** 2 / (b + c)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "chi2_cc": float(chi2), "p_value": p}


def paired_bootstrap(y, probs_a: Sequence[np.ndarray], probs_b: Sequence[np.ndarray],
                     metric_fn: Callable, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the difference mean_seeds(metric(A)) - mean_seeds(metric(B)) over test transactions."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    obs = np.mean([metric_fn(y, p) for p in probs_a]) - np.mean([metric_fn(y, p) for p in probs_b])
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = (np.mean([metric_fn(y[idx], p[idx]) for p in probs_a])
                - np.mean([metric_fn(y[idx], p[idx]) for p in probs_b]))
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"diff": float(obs), "ci_low": float(np.quantile(d, .025)), "ci_high": float(np.quantile(d, .975)),
            "p_value": p}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def average_ranks(scores: pd.DataFrame, higher_is_better: bool = True) -> pd.Series:
    """scores: rows = blocks, columns = models. Rank 1 = best; ties get the average rank."""
    r = scores.rank(axis=1, ascending=not higher_is_better, method="average")
    return r.mean(axis=0).sort_values()


def friedman_nemenyi(scores: pd.DataFrame, higher_is_better: bool = True, alpha: float = 0.05) -> dict:
    """Friedman chi-square over blocks and the Nemenyi critical difference of average ranks."""
    k, n = scores.shape[1], scores.shape[0]
    chi2, p = stats.friedmanchisquare(*[scores[c].to_numpy() for c in scores.columns])
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"avg_ranks": average_ranks(scores, higher_is_better), "friedman_chi2": float(chi2),
            "friedman_p": float(p), "cd": float(cd), "k": k, "n_blocks": n, "alpha": alpha}


def rank_table(df: pd.DataFrame, directions: Dict[str, int]) -> pd.DataFrame:
    """Rank of each model (rows) on each metric (columns); 1 = best."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = df[c].rank(ascending=directions.get(c, 1) < 0, method="min").astype(int)
    return out
