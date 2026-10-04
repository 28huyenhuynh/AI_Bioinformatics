"""Paired significance testing across subjects.

Each subject gives one score per model, so models are compared with the
Wilcoxon signed-rank test on paired per-subject scores (non-parametric, no
normality assumption, n = 27 pairs). Many pairwise tests are run, so p-values
are corrected with Holm-Bonferroni.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(a, b):
    """Effect size for Wilcoxon: +1 = a always better, -1 = b always better."""
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()


def pairwise_wilcoxon(df, metric="macro_f1", model_col="model", subject_col="subject"):
    wide = df.pivot_table(index=subject_col, columns=model_col, values=metric).dropna()
    rows = []
    for a, b in combinations(wide.columns, 2):
        x, y = wide[a].to_numpy(), wide[b].to_numpy()
        if np.allclose(x, y):
            p = 1.0
        else:
            p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue
        rows.append({"model_a": a, "model_b": b, "n_subjects": len(x),
                     f"median_{metric}_a": np.median(x), f"median_{metric}_b": np.median(y),
                     "median_diff_a_minus_b": np.median(x - y),
                     "rank_biserial": rank_biserial(x, y), "p_value": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_value"])
        out["significant_0.05"] = out["p_holm"] < 0.05
    return out


def summary_table(df, metrics=("accuracy", "balanced_accuracy", "macro_f1")):
    g = df.groupby("model")[list(metrics)]
    mean, std = g.mean(), g.std()
    out = pd.DataFrame(index=mean.index)
    for m in metrics:
        out[m] = [f"{mu:.3f} ± {sd:.3f}" for mu, sd in zip(mean[m], std[m])]
    out["n_subjects"] = df.groupby("model")["subject"].nunique()
    return out.loc[mean[list(metrics)[-1]].sort_values(ascending=False).index]
