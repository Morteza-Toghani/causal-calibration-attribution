"""Statistical inference functions — FROZEN semantics.

Bootstrap, permutation, and t-based CI at the seed level (n = number of
independent training seeds).
"""
from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(diffs, n_boot: int = 5000, seed: int = 42):
    """Percentile bootstrap CI for the mean of paired differences.

    Returns (mean, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(low), float(high)


def t_ci(values, alpha: float = 0.05):
    """Student-t CI for the mean of a small sample.

    df = n - 1. Correct for n = 5 where bootstrap percentile is too narrow.
    Returns (mean, lo, hi).
    """
    from scipy.stats import t as t_dist
    x = np.asarray(values, dtype=float)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    t_crit = float(t_dist.ppf(1 - alpha / 2, df=n - 1))
    return m, m - t_crit * se, m + t_crit * se


def paired_permutation_pvalue(diffs, n_perm: int = 10000, seed: int = 42):
    """Exact sign-flip permutation p-value for the paired null.

    Note: with n = 5 seeds the minimum attainable two-sided p is 0.0625.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    observed = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)))
    perm_means = np.abs((signs * d).mean(axis=1))
    return float((1.0 + np.sum(perm_means >= observed)) / (n_perm + 1.0))
