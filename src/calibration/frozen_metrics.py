"""FROZEN metric_bundle — exact copy from Paper1 pipeline.

This module is the single source of truth for the calibration and
uncertainty metrics used in the paper. It replaces the inline
implementations that previously lived in individual scripts.
"""
from __future__ import annotations

import math

import numpy as np

NOMINAL_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def z_for_central_coverage(c: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(0.5 + c / 2.0))


def regression_calibration_curve(mu, var, y, levels=NOMINAL_LEVELS):
    """Return (curve, ace) where curve is a list of {nominal, empirical} and
    ace is the mean absolute deviation across levels."""
    sigma = np.sqrt(var)
    curve = []
    errors = []
    for c in levels:
        z = z_for_central_coverage(c)
        hw = z * sigma
        inside = (y >= mu - hw) & (y <= mu + hw)
        emp = float(inside.mean())
        curve.append({'nominal': float(c), 'empirical': emp})
        errors.append(abs(emp - c))
    return curve, float(np.mean(errors))


def metric_bundle(pred: dict, y: np.ndarray) -> dict:
    """Return the FROZEN 7-metric bundle + calibration curve.

    Expected `pred` keys: mu, var.
    """
    from scipy.special import ndtr
    from scipy.stats import norm

    mu = pred['mu']
    var = pred['var']
    sigma = np.sqrt(var)

    # Moment-matched Gaussian predictive NLL
    logvar = np.log(var)
    nll = 0.5 * (logvar + (y - mu) ** 2 / var + math.log(2.0 * math.pi))
    nll = float(nll.mean())

    curve, ace = regression_calibration_curve(mu, var, y)

    c50 = next(x['empirical'] for x in curve if math.isclose(x['nominal'], 0.5))
    c90 = next(x['empirical'] for x in curve if math.isclose(x['nominal'], 0.9))

    z50 = z_for_central_coverage(0.5)
    z90 = z_for_central_coverage(0.9)
    sharp50 = float(np.mean(2.0 * z50 * sigma))
    sharp90 = float(np.mean(2.0 * z90 * sigma))

    # Closed-form CRPS for a Gaussian
    z = (y - mu) / sigma
    crps = sigma * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )
    crps = float(np.mean(crps))

    return {
        'calibration_error': ace,
        'nll': nll,
        'coverage_50': float(c50),
        'coverage_90': float(c90),
        'sharpness_50': sharp50,
        'sharpness_90': sharp90,
        'crps': crps,
        'calibration_curve': curve,
    }
