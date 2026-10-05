"""Regression calibration metrics for Gaussian predictive distributions.

Definitions (all per-element, then aggregated):

* Central prediction interval at nominal level ``p``: ``mean +/- z * std`` with
  ``z = Phi^{-1}((1 + p) / 2)``.
* Empirical coverage: fraction of targets inside that interval.
* ``calibration_error`` (primary): mean over nominal levels of
  ``| coverage_pooled(p) - p |`` where ``coverage_pooled`` is the coverage
  averaged over samples AND dimensions. This is the definition that reproduces
  the ``calibration_error`` values stored in ``results/raw`` (verified in
  ``scripts/analyze_results.py``).
* ``calibration_error_macro``: same, but the absolute deviation is taken per
  dimension before averaging (cannot be masked by dimensions with opposite sign).
* ``signed_calibration_bias``: mean over levels of ``coverage_pooled(p) - p``.
  Positive = over-coverage (intervals too wide), negative = under-coverage.
  The primary metric is unsigned, so it cannot distinguish the two.

Shapes: ``y``, ``mean``, ``var`` are ``(N, D)`` arrays (a ``(N,)`` array is
treated as ``D = 1``).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

DEFAULT_LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 10)  # 0.1 ... 0.9


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def _check(y, mean, var):
    y, mean, var = _as_2d(y), _as_2d(mean), _as_2d(var)
    if not (y.shape == mean.shape == var.shape):
        raise ValueError(f"shape mismatch: y{y.shape} mean{mean.shape} var{var.shape}")
    if np.any(var <= 0):
        raise ValueError("predictive variance must be strictly positive")
    return y, mean, var


def interval_halfwidth(var: np.ndarray, level: float) -> np.ndarray:
    """Half-width of the central Gaussian interval with nominal coverage ``level``."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    return norm.ppf(0.5 + level / 2.0) * np.sqrt(var)


def coverage_per_dim(y, mean, var, levels=DEFAULT_LEVELS) -> np.ndarray:
    """Empirical coverage, shape ``(len(levels), D)``."""
    y, mean, var = _check(y, mean, var)
    abs_err = np.abs(y - mean)
    out = np.empty((len(levels), y.shape[1]))
    for i, p in enumerate(levels):
        out[i] = (abs_err <= interval_halfwidth(var, p)).mean(axis=0)
    return out


def calibration_curve(y, mean, var, levels=DEFAULT_LEVELS) -> dict:
    """Nominal vs pooled empirical coverage (plus per-dimension coverages)."""
    cov = coverage_per_dim(y, mean, var, levels)
    pooled = cov.mean(axis=1)
    levels = np.asarray(levels, dtype=float)
    return {
        "nominal": levels,
        "empirical": pooled,
        "abs_error": np.abs(pooled - levels),
        "dimension_coverages": cov,
    }


def calibration_error(y, mean, var, levels=DEFAULT_LEVELS) -> float:
    """Primary metric: mean_p | pooled coverage(p) - p |."""
    c = calibration_curve(y, mean, var, levels)
    return float(c["abs_error"].mean())


def calibration_error_macro(y, mean, var, levels=DEFAULT_LEVELS) -> float:
    """Mean over levels and dimensions of | coverage_d(p) - p |."""
    cov = coverage_per_dim(y, mean, var, levels)
    return float(np.abs(cov - np.asarray(levels)[:, None]).mean())


def signed_calibration_bias(y, mean, var, levels=DEFAULT_LEVELS) -> float:
    """Mean_p ( pooled coverage(p) - p ). >0 over-coverage, <0 under-coverage."""
    c = calibration_curve(y, mean, var, levels)
    return float((c["empirical"] - c["nominal"]).mean())


def coverage_at(y, mean, var, level: float) -> float:
    """Pooled empirical coverage at a single nominal level."""
    return float(coverage_per_dim(y, mean, var, [level]).mean())


def sharpness(var, level: float) -> float:
    """Mean width of the central interval at ``level`` (smaller = sharper)."""
    var = _as_2d(var)
    return float((2.0 * interval_halfwidth(var, level)).mean())


def gaussian_nll(y, mean, var) -> float:
    """Mean per-element Gaussian negative log-likelihood (raw units)."""
    y, mean, var = _check(y, mean, var)
    return float((0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var).mean())


def gaussian_crps(y, mean, var) -> float:
    """Mean closed-form CRPS of a Gaussian predictive distribution."""
    y, mean, var = _check(y, mean, var)
    sd = np.sqrt(var)
    z = (y - mean) / sd
    crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(crps.mean())
