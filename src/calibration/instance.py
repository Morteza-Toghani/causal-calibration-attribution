"""Instance-level calibration error — FROZEN pipeline semantics.

Matches the instance_calibration_error helper that appears twice
inside train_ensemble_for_seed in the FROZEN pipeline.

For each sample i and each nominal level c:
    cov_i_c = mean_d inside(y_i_d in [mu_i_d +/- z_c * sqrt(var_i_d)])
    err_i   = mean_c |cov_i_c - c|

Returns a vector (N,) of per-instance calibration errors.

Note: this differs from the aggregate calibration_error in
src/calibration/regression.py. The aggregate first averages coverage
across samples then takes abs; this one takes abs per sample then
averages. The two are NOT equal (Jensen's inequality). The FROZEN
pipeline uses this instance-level version for paired ATE.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

NOMINAL_COVERAGES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def z_for_central_coverage(c: float) -> float:
    return float(norm.ppf(0.5 + c / 2.0))


def instance_calibration_error(
    pred_mu: np.ndarray,
    pred_var: np.ndarray,
    y: np.ndarray,
    levels=NOMINAL_COVERAGES,
) -> np.ndarray:
    """Return shape (N,) per-instance calibration error."""
    errors = []
    for c in levels:
        zc = z_for_central_coverage(c)
        hw = zc * np.sqrt(pred_var)
        inside = ((y >= pred_mu - hw) & (y <= pred_mu + hw)).mean(axis=1)
        errors.append(np.abs(inside - c))
    return np.mean(np.stack(errors, axis=1), axis=1)


def aggregate_calibration_error(
    pred_mu: np.ndarray,
    pred_var: np.ndarray,
    y: np.ndarray,
    levels=NOMINAL_COVERAGES,
) -> float:
    """Aggregate (macro) calibration error, for comparison."""
    errors = []
    for c in levels:
        zc = z_for_central_coverage(c)
        hw = zc * np.sqrt(pred_var)
        inside = (y >= pred_mu - hw) & (y <= pred_mu + hw)
        emp = float(inside.mean())
        errors.append(abs(emp - c))
    return float(np.mean(errors))
