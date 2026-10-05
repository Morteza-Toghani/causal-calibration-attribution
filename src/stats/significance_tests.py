"""Paired significance testing and multiple-comparison correction."""

from __future__ import annotations

import itertools

import numpy as np


def signflip_permutation_pvalue(
    effects, n_resamples: int = 100_000, seed: int = 0, exact_max_n: int = 20
) -> float:
    """Two-sided paired sign-flip permutation test of H0: effects are symmetric about 0.

    Statistic: |mean(effects)|. Exact enumeration of all 2^n sign patterns when
    ``n <= exact_max_n``, otherwise Monte-Carlo with the +1 correction.
    NOTE: with n units the smallest attainable exact two-sided p is 2 / 2^n
    (n = 5 -> 0.0625), so n = 5 seeds can never reach p < 0.05.
    """
    d = np.asarray(effects, float)
    n = d.size
    if n < 2:
        raise ValueError("need at least 2 values")
    obs = abs(d.mean())
    tol = 1e-12 * max(1.0, obs)
    if n <= exact_max_n:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        stat = np.abs((signs * d).mean(axis=1))
        return float((stat >= obs - tol).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    stat = np.abs((signs * d).mean(axis=1))
    return float(((stat >= obs - tol).sum() + 1) / (n_resamples + 1))


def min_attainable_pvalue(n: int) -> float:
    """Smallest two-sided exact sign-flip p-value with n paired units."""
    return 2.0 / 2**n


def holm_adjust(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvalues, float)
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj
