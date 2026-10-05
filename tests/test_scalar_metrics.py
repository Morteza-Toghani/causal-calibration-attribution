import numpy as np
import pytest
from scipy.stats import norm

from src.calibration import scalar_metrics as cal

RNG = np.random.default_rng(123)
N, D = 200_000, 3


def _data(pred_var, true_var=1.0):
    y = RNG.normal(0, np.sqrt(true_var), size=(N, D))
    return y, np.zeros((N, D)), np.full((N, D), pred_var)


def test_calibrated_gaussian_has_small_error_and_bias():
    y, m, v = _data(1.0)
    assert cal.calibration_error(y, m, v) < 0.003
    assert abs(cal.signed_calibration_bias(y, m, v)) < 0.003
    assert cal.calibration_error_macro(y, m, v) < 0.004


def test_overdispersed_prediction_overcovers():
    y, m, v = _data(4.0)  # predicted var 4x too large
    assert cal.signed_calibration_bias(y, m, v) > 0.1
    assert cal.coverage_at(y, m, v, 0.5) > 0.8


def test_underdispersed_prediction_undercovers():
    y, m, v = _data(0.25)
    assert cal.signed_calibration_bias(y, m, v) < -0.1
    assert cal.coverage_at(y, m, v, 0.9) < 0.8


def test_unsigned_error_equals_abs_signed_bias_when_sign_is_constant():
    y, m, v = _data(4.0)
    assert cal.calibration_error(y, m, v) == pytest.approx(abs(cal.signed_calibration_bias(y, m, v)))


def test_pooled_error_can_be_masked_but_macro_error_cannot():
    # dim 0 over-covers, dim 1 under-covers: opposite signs partially cancel in the pooled metric
    y = RNG.normal(size=(N, 2))
    m = np.zeros((N, 2))
    v = np.stack([np.full(N, 4.0), np.full(N, 0.25)], axis=1)
    pooled, macro = cal.calibration_error(y, m, v), cal.calibration_error_macro(y, m, v)
    assert macro >= pooled  # triangle inequality always holds
    assert macro > 3 * pooled  # ... and here cancellation makes the gap large


def test_exact_quantile_interval_hits_nominal_coverage():
    y, m, v = _data(1.0)
    assert cal.coverage_at(y, m, v, 0.5) == pytest.approx(0.5, abs=0.005)
    assert cal.coverage_at(y, m, v, 0.9) == pytest.approx(0.9, abs=0.005)


def test_curve_shapes_and_abs_error_consistency():
    y, m, v = _data(1.0)
    c = cal.calibration_curve(y, m, v)
    assert c["dimension_coverages"].shape == (9, D)
    assert np.allclose(c["abs_error"], np.abs(c["empirical"] - c["nominal"]))


def test_nll_of_standard_normal_at_mean():
    assert cal.gaussian_nll(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1))) == pytest.approx(
        0.5 * np.log(2 * np.pi)
    )


def test_crps_of_standard_normal_at_mean():
    expected = 2 * norm.pdf(0) - 1 / np.sqrt(np.pi)
    assert cal.gaussian_crps(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1))) == pytest.approx(expected)


def test_sharpness_scales_with_sigma():
    w1 = cal.sharpness(np.ones((10, 2)), 0.9)
    w2 = cal.sharpness(np.full((10, 2), 4.0), 0.9)
    assert w2 == pytest.approx(2 * w1)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        cal.calibration_error(np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)))  # var = 0
    with pytest.raises(ValueError):
        cal.calibration_error(np.zeros((3, 1)), np.zeros((4, 1)), np.ones((3, 1)))
    with pytest.raises(ValueError):
        cal.interval_halfwidth(np.ones(3), 1.0)
