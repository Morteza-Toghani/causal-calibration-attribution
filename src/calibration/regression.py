from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import norm


EPS = 1e-8


def validate_shapes(
    y_true: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> None:
    """
    Validate expected shapes.

    y_true:
        [N, D]

    means:
        [K, N, D]

    variances:
        [K, N, D]
    """
    y_true = np.asarray(y_true)
    means = np.asarray(means)
    variances = np.asarray(variances)

    if y_true.ndim != 2:
        raise ValueError(
            f"y_true must have shape [N, D], got {y_true.shape}"
        )

    if means.ndim != 3:
        raise ValueError(
            f"means must have shape [K, N, D], got {means.shape}"
        )

    if variances.ndim != 3:
        raise ValueError(
            f"variances must have shape [K, N, D], got {variances.shape}"
        )

    if means.shape != variances.shape:
        raise ValueError(
            f"means and variances must have identical shape, "
            f"got {means.shape} vs {variances.shape}"
        )

    k, n, d = means.shape

    if y_true.shape != (n, d):
        raise ValueError(
            f"y_true shape {y_true.shape} does not match "
            f"ensemble shape [N, D] = {(n, d)}"
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")

    if not np.all(np.isfinite(means)):
        raise ValueError("means contains NaN or infinite values.")

    if not np.all(np.isfinite(variances)):
        raise ValueError("variances contains NaN or infinite values.")

    if np.any(variances <= 0):
        raise ValueError("All predictive variances must be strictly positive.")


def ensemble_moments(
    member_means: np.ndarray,
    member_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ensemble predictive mean and total variance.

    For ensemble member k:

        p_k(y) = Normal(mu_k, sigma_k^2)

    Ensemble mean:

        mu = mean_k(mu_k)

    Total variance:

        Var_total =
            mean_k(sigma_k^2)
            + Var_k(mu_k)

    Expected input shape:
        [K, N, D]

    Returns:
        predictive_mean: [N, D]
        predictive_variance: [N, D]
    """
    member_means = np.asarray(member_means, dtype=np.float64)
    member_variances = np.asarray(member_variances, dtype=np.float64)

    if member_means.shape != member_variances.shape:
        raise ValueError("Member means and variances must have the same shape.")

    if member_means.ndim != 3:
        raise ValueError(
            f"Expected [K, N, D], got {member_means.shape}"
        )

    if np.any(member_variances <= 0):
        raise ValueError("Member variances must be strictly positive.")

    predictive_mean = np.mean(member_means, axis=0)

    aleatoric_variance = np.mean(member_variances, axis=0)
    ensemble_variance = np.var(member_means, axis=0)

    predictive_variance = (
        aleatoric_variance + ensemble_variance
    )

    predictive_variance = np.maximum(
        predictive_variance,
        EPS,
    )

    return predictive_mean, predictive_variance


def prediction_interval(
    mean: np.ndarray,
    variance: np.ndarray,
    nominal_coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a central Gaussian prediction interval.

    nominal_coverage:
        e.g. 0.50 or 0.90
    """
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError(
            "nominal_coverage must be between 0 and 1."
        )

    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)

    if np.any(variance <= 0):
        raise ValueError("Variance must be strictly positive.")

    std = np.sqrt(np.maximum(variance, EPS))

    alpha = 1.0 - nominal_coverage
    z = norm.ppf(1.0 - alpha / 2.0)

    lower = mean - z * std
    upper = mean + z * std

    return lower, upper


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Empirical coverage per output dimension.

    Returns:
        [D]
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    covered = (
        (y_true >= lower)
        & (y_true <= upper)
    )

    return np.mean(covered, axis=0)


def interval_sharpness(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Mean prediction-interval width per output dimension.

    Smaller intervals are sharper, provided calibration is maintained.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    if np.any(upper < lower):
        raise ValueError("Upper interval bound cannot be below lower bound.")

    return np.mean(upper - lower, axis=0)


def gaussian_nll(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    """
    Gaussian negative log-likelihood per output dimension.

    Returns:
        [D]
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)

    variance = np.maximum(variance, EPS)
    std = np.sqrt(variance)

    log_density = norm.logpdf(
        y_true,
        loc=mean,
        scale=std,
    )

    return np.mean(-log_density, axis=0)


def gaussian_crps(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    """
    CRPS for a Gaussian predictive distribution.

    Returns:
        [D]
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)

    variance = np.maximum(variance, EPS)
    std = np.sqrt(variance)

    z = (y_true - mean) / std

    phi = norm.pdf(z)
    Phi = norm.cdf(z)

    crps = std * (
        z * (2.0 * Phi - 1.0)
        + 2.0 * phi
        - 1.0 / np.sqrt(np.pi)
    )

    return np.mean(crps, axis=0)


def calibration_error(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    nominal_levels: Iterable[float] = (0.50, 0.90),
) -> dict:
    """
    Regression calibration based on interval coverage.

    For each nominal level:

        coverage_error =
            empirical_coverage - nominal_coverage

        coverage_deviation =
            abs(coverage_error)

    Overall calibration error is the macro-average of absolute
    deviations across nominal levels and output dimensions.
    """
    nominal_levels = tuple(nominal_levels)

    per_level = {}

    deviations = []

    for nominal in nominal_levels:
        lower, upper = prediction_interval(
            mean,
            variance,
            nominal,
        )

        coverage = interval_coverage(
            y_true,
            lower,
            upper,
        )

        signed_error = coverage - nominal
        absolute_error = np.abs(signed_error)

        deviations.append(absolute_error)

        per_level[str(nominal)] = {
            "nominal_coverage": float(nominal),
            "empirical_coverage_mean": float(
                np.mean(coverage)
            ),
            "coverage_error_mean": float(
                np.mean(signed_error)
            ),
            "coverage_deviation_mean": float(
                np.mean(absolute_error)
            ),
            "coverage_per_dimension": coverage.tolist(),
            "coverage_error_per_dimension": signed_error.tolist(),
            "sharpness_mean": float(
                np.mean(
                    interval_sharpness(
                        lower,
                        upper,
                    )
                )
            ),
            "sharpness_per_dimension": (
                interval_sharpness(
                    lower,
                    upper,
                ).tolist()
            ),
        }

    calibration_error_value = float(
        np.mean(
            np.stack(deviations, axis=0)
        )
    )

    return {
        "calibration_error": calibration_error_value,
        "levels": per_level,
    }


def evaluate_regression_calibration(
    y_true: np.ndarray,
    member_means: np.ndarray,
    member_variances: np.ndarray,
    nominal_levels: Iterable[float] = (0.50, 0.90),
) -> dict:
    """
    Full Phase 3 regression-calibration evaluation.

    Returns per-dimension and macro-averaged metrics.
    """
    validate_shapes(
        y_true,
        member_means,
        member_variances,
    )

    mean, variance = ensemble_moments(
        member_means,
        member_variances,
    )

    calibration = calibration_error(
        y_true,
        mean,
        variance,
        nominal_levels,
    )

    nll_per_dimension = gaussian_nll(
        y_true,
        mean,
        variance,
    )

    crps_per_dimension = gaussian_crps(
        y_true,
        mean,
        variance,
    )

    return {
        "n_samples": int(y_true.shape[0]),
        "n_dimensions": int(y_true.shape[1]),
        "n_ensemble_members": int(member_means.shape[0]),
        "calibration_error": calibration[
            "calibration_error"
        ],
        "nll_mean": float(
            np.mean(nll_per_dimension)
        ),
        "crps_mean": float(
            np.mean(crps_per_dimension)
        ),
        "nll_per_dimension": nll_per_dimension.tolist(),
        "crps_per_dimension": crps_per_dimension.tolist(),
        "predictive_variance_mean": float(
            np.mean(variance)
        ),
        "predictive_std_mean": float(
            np.mean(np.sqrt(variance))
        ),
        "calibration": calibration,
    }
