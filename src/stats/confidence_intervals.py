"""Confidence intervals for the mean of per-seed effects."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_mean_ci(values, level: float = 0.95, n_resamples: int = 10_000, seed: int = 0):
    """Percentile bootstrap CI for the mean. Resampling unit = one value (e.g. a seed).

    With very few units (n ~ 5) percentile bootstrap intervals are optimistic;
    report ``t_mean_ci`` next to it.
    """
    v = np.asarray(values, float)
    if v.size < 2:
        raise ValueError("need at least 2 values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(lo), float(hi)


def t_mean_ci(values, level: float = 0.95):
    """Student-t CI for the mean."""
    v = np.asarray(values, float)
    if v.size < 2:
        raise ValueError("need at least 2 values")
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
    h = stats.t.ppf(0.5 + level / 2, df=v.size - 1) * se
    return float(m - h), float(m + h)
