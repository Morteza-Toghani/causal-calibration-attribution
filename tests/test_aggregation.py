import numpy as np
import pytest

from src.model.aggregation import aggregate_gaussian_ensemble, aleatoric_fraction


def test_identical_members_have_zero_epistemic_variance():
    mus = np.ones((5, 4, 2))
    var = np.full((5, 4, 2), 0.3)
    agg = aggregate_gaussian_ensemble(mus, var)
    assert np.allclose(agg["epistemic_var"], 0.0)
    assert np.allclose(agg["total_var"], 0.3)
    assert aleatoric_fraction(agg) == pytest.approx(1.0)


def test_total_variance_equals_mixture_variance():
    rng = np.random.default_rng(0)
    K, N, D = 5, 3, 2
    mus = rng.normal(size=(K, N, D))
    var = rng.uniform(0.1, 1.0, size=(K, N, D))
    agg = aggregate_gaussian_ensemble(mus, var)
    # second moment of the mixture minus squared mean
    mix_var = (var + mus**2).mean(axis=0) - mus.mean(axis=0) ** 2
    assert np.allclose(agg["total_var"], mix_var)
    assert np.allclose(agg["mean"], mus.mean(axis=0))


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        aggregate_gaussian_ensemble(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        aggregate_gaussian_ensemble(np.ones((2, 3, 1)), np.zeros((2, 3, 1)))
