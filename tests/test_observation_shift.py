import numpy as np
import pytest

from src.interventions.observation import apply_observation_shift, draw_base_noise

OBS = np.random.default_rng(1).normal(size=(50, 11))
SCALE = np.full(11, 2.0)


def test_zero_intensity_is_identity_null_intervention():
    eps = draw_base_noise(OBS.shape, seed=7)
    assert np.array_equal(apply_observation_shift(OBS, eps, 0.0, SCALE), OBS)


def test_base_noise_is_deterministic_per_seed():
    assert np.array_equal(draw_base_noise((4, 3), 3), draw_base_noise((4, 3), 3))
    assert not np.array_equal(draw_base_noise((4, 3), 3), draw_base_noise((4, 3), 4))


def test_crn_high_perturbation_is_exact_rescaling_of_low():
    eps = draw_base_noise(OBS.shape, seed=7)
    low = apply_observation_shift(OBS, eps, 0.05, SCALE) - OBS
    high = apply_observation_shift(OBS, eps, 0.10, SCALE) - OBS
    assert np.allclose(high, 2 * low)


def test_shape_mismatch_and_negative_intensity_raise():
    with pytest.raises(ValueError):
        apply_observation_shift(OBS, np.zeros((3, 3)), 0.1, SCALE)
    with pytest.raises(ValueError):
        apply_observation_shift(OBS, np.zeros_like(OBS), -0.1, SCALE)
