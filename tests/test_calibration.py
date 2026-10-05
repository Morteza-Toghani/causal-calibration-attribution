"""
Phase 3 — Unit tests for regression calibration metrics.

These tests validate the SEMANTICS of the calibration metrics on synthetic
data with known ground-truth properties, per the master plan's requirement
(Section 19, Sanity Checks): "Calibration implementation audit: coverage/
binning should be verified on synthetic data with known calibration via
unit tests."
"""

import numpy as np
import pytest

from src.calibration.regression import (
    ensemble_moments,
    evaluate_regression_calibration,
    validate_shapes,
)

# ---------------------------------------------------------------------------
# 1. Perfectly (near-perfectly) calibrated Gaussian
# ---------------------------------------------------------------------------

def test_interval_coverage_of_standard_normal():
    """If y ~ N(0, 1) and we predict N(0, 1), empirical coverage at the
    50% and 90% nominal levels should match nominal levels closely."""
    rng = np.random.default_rng(123)

    y = rng.normal(loc=0.0, scale=1.0, size=(50_000, 1))
    means = np.zeros((5, 50_000, 1))
    variances = np.ones((5, 50_000, 1))

    result = evaluate_regression_calibration(
        y_true=y,
        member_means=means,
        member_variances=variances,
    )

    assert abs(
        result["calibration"]["levels"]["0.5"]["empirical_coverage_mean"] - 0.50
    ) < 0.02

    assert abs(
        result["calibration"]["levels"]["0.9"]["empirical_coverage_mean"] - 0.90
    ) < 0.02

    # A well-calibrated model should have low calibration error
    assert result["calibration_error"] < 0.02


# ---------------------------------------------------------------------------
# 2. Overly narrow intervals (underconfident coverage -> undercoverage)
# ---------------------------------------------------------------------------

def test_overly_narrow_intervals_undercover():
    """If predicted variance is much smaller than true variance, empirical
    coverage should fall BELOW nominal coverage (intervals too narrow)."""
    rng = np.random.default_rng(7)

    true_std = 2.0
    y = rng.normal(loc=0.0, scale=true_std, size=(50_000, 1))

    # Model claims std=0.5, much narrower than the true std=2.0
    means = np.zeros((5, 50_000, 1))
    variances = np.full((5, 50_000, 1), 0.5 ** 2)

    result = evaluate_regression_calibration(
        y_true=y,
        member_means=means,
        member_variances=variances,
    )

    cov_90 = result["calibration"]["levels"]["0.9"]["empirical_coverage_mean"]
    assert cov_90 < 0.90  # under-covers
    assert result["calibration"]["levels"]["0.9"]["coverage_error_mean"] < 0
    assert result["calibration_error"] > 0.05


# ---------------------------------------------------------------------------
# 3. Overly wide intervals (overconfident coverage -> overcoverage)
# ---------------------------------------------------------------------------

def test_overly_wide_intervals_overcover():
    """If predicted variance is much larger than true variance, empirical
    coverage should exceed nominal coverage (intervals too wide)."""
    rng = np.random.default_rng(42)

    true_std = 0.5
    y = rng.normal(loc=0.0, scale=true_std, size=(50_000, 1))

    # Model claims std=2.0, much wider than the true std=0.5
    means = np.zeros((5, 50_000, 1))
    variances = np.full((5, 50_000, 1), 2.0 ** 2)

    result = evaluate_regression_calibration(
        y_true=y,
        member_means=means,
        member_variances=variances,
    )

    cov_90 = result["calibration"]["levels"]["0.9"]["empirical_coverage_mean"]
    assert cov_90 > 0.90  # over-covers
    assert result["calibration"]["levels"]["0.9"]["coverage_error_mean"] > 0
    assert result["calibration_error"] > 0.05


# ---------------------------------------------------------------------------
# 4. Zero / near-zero numerical variance handling
# ---------------------------------------------------------------------------

def test_zero_variance_is_clamped_not_crashing():
    """Member variances of exactly zero should be rejected by
    ensemble_moments's input validation (variances must be > 0), since a
    literal zero predictive variance is not a valid Gaussian and should be
    caught explicitly rather than silently clamped upstream."""
    means = np.zeros((5, 10, 1))
    variances = np.zeros((5, 10, 1))
    y = np.zeros((10, 1))

    with pytest.raises(ValueError):
        evaluate_regression_calibration(
            y_true=y,
            member_means=means,
            member_variances=variances,
        )


def test_near_zero_variance_does_not_produce_nan_or_inf():
    """Very small (but positive) variances should be handled gracefully via
    the EPS floor, without producing NaN/Inf in NLL or CRPS."""
    rng = np.random.default_rng(1)
    means = np.zeros((5, 100, 1))
    variances = np.full((5, 100, 1), 1e-12)
    y = rng.normal(0, 1e-6, size=(100, 1))

    result = evaluate_regression_calibration(
        y_true=y,
        member_means=means,
        member_variances=variances,
    )

    assert np.isfinite(result["nll_mean"])
    assert np.isfinite(result["crps_mean"])
    assert np.isfinite(result["calibration_error"])


# ---------------------------------------------------------------------------
# 5. Shape validation
# ---------------------------------------------------------------------------

def test_shape_validation_rejects_bad_y_true_ndim():
    y_true = np.zeros(10)  # wrong: should be [N, D]
    means = np.zeros((5, 10, 1))
    variances = np.ones((5, 10, 1))
    with pytest.raises(ValueError):
        validate_shapes(y_true, means, variances)


def test_shape_validation_rejects_mismatched_means_variances():
    y_true = np.zeros((10, 1))
    means = np.zeros((5, 10, 1))
    variances = np.ones((5, 10, 2))  # mismatched D
    with pytest.raises(ValueError):
        validate_shapes(y_true, means, variances)


def test_shape_validation_rejects_y_true_n_d_mismatch():
    y_true = np.zeros((9, 1))  # N=9 but ensemble has N=10
    means = np.zeros((5, 10, 1))
    variances = np.ones((5, 10, 1))
    with pytest.raises(ValueError):
        validate_shapes(y_true, means, variances)


def test_shape_validation_rejects_nan():
    y_true = np.zeros((10, 1))
    y_true[0, 0] = np.nan
    means = np.zeros((5, 10, 1))
    variances = np.ones((5, 10, 1))
    with pytest.raises(ValueError):
        validate_shapes(y_true, means, variances)


# ---------------------------------------------------------------------------
# 6. Ensemble moments sanity check (matches the master plan's formula)
# ---------------------------------------------------------------------------

def test_ensemble_moments_matches_manual_formula():
    """Var_total = mean_k(sigma_k^2) + Var_k(mu_k), checked against a
    hand-computed small example."""
    # 3 members, 2 samples, 1 dimension
    member_means = np.array([
        [[1.0], [2.0]],
        [[3.0], [2.0]],
        [[5.0], [2.0]],
    ])  # shape (3, 2, 1)
    member_variances = np.array([
        [[0.1], [0.2]],
        [[0.1], [0.2]],
        [[0.1], [0.2]],
    ])  # shape (3, 2, 1)

    mean, var = ensemble_moments(member_means, member_variances)

    # sample 0: means = [1,3,5] -> mean=3, var=population var of [1,3,5]
    expected_mean_0 = np.mean([1.0, 3.0, 5.0])
    expected_aleatoric_0 = np.mean([0.1, 0.1, 0.1])
    expected_epistemic_0 = np.var([1.0, 3.0, 5.0])  # population variance
    expected_var_0 = expected_aleatoric_0 + expected_epistemic_0

    assert np.isclose(mean[0, 0], expected_mean_0)
    assert np.isclose(var[0, 0], expected_var_0)

    # sample 1: all members agree (mu_k=2 for all k) -> epistemic var = 0
    assert np.isclose(mean[1, 0], 2.0)
    assert np.isclose(var[1, 0], 0.2, atol=1e-6)  # only aleatoric remains
