import numpy as np
import pandas as pd
import pytest

from src.stats.confidence_intervals import bootstrap_mean_ci, t_mean_ci
from src.stats.effect_sizes import paired_cohens_dz, sign_consistency
from src.stats.multi_seed_runner import summarize_seed_effects
from src.stats.significance_tests import (
    holm_adjust,
    min_attainable_pvalue,
    signflip_permutation_pvalue,
)


def test_holm_known_example():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_monotone_and_capped():
    adj = holm_adjust([0.5, 0.9, 0.001, 0.2])
    assert np.all(adj <= 1.0) and np.all(adj >= np.array([0.5, 0.9, 0.001, 0.2]))


def test_exact_signflip_floor_for_five_seeds():
    p = signflip_permutation_pvalue([0.1, 0.2, 0.3, 0.4, 0.5])
    assert p == pytest.approx(2 / 32) == pytest.approx(min_attainable_pvalue(5))


def test_signflip_symmetric_effects_give_p_one():
    assert signflip_permutation_pvalue([-1.0, 1.0, -2.0, 2.0]) == pytest.approx(1.0)


def test_signflip_large_n_detects_shift_and_respects_null():
    rng = np.random.default_rng(0)
    assert signflip_permutation_pvalue(rng.normal(1.0, 1.0, 40), n_resamples=5000) < 0.01
    assert signflip_permutation_pvalue(rng.normal(0.0, 1.0, 40), n_resamples=5000) > 0.05


def test_cohens_dz_and_degenerate_cases():
    d = np.array([1.0, 2.0, 3.0])
    assert paired_cohens_dz(d) == pytest.approx(2.0)
    assert np.isnan(paired_cohens_dz([1.0, 1.0, 1.0]))
    assert np.isnan(paired_cohens_dz([1.0]))
    assert sign_consistency([1, -1, 2]) == (2, 3)


def test_bootstrap_ci_brackets_mean_and_is_reproducible():
    v = np.array([0.9, 1.1, 1.0, 1.2, 0.8])
    lo, hi = bootstrap_mean_ci(v, seed=1)
    assert lo < v.mean() < hi
    assert (lo, hi) == bootstrap_mean_ci(v, seed=1)


def test_t_ci_matches_scipy_formula():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo, hi = t_mean_ci(v)
    assert lo == pytest.approx(3 - 2.7764451 * (np.sqrt(2.5) / np.sqrt(5)), abs=1e-5)
    assert hi == pytest.approx(3 + 2.7764451 * (np.sqrt(2.5) / np.sqrt(5)), abs=1e-5)


def test_summarize_seed_effects_columns_and_holm():
    rows = []
    for cond, mech, inten, vals in [
        ("dynamics_high", "dynamics", "high", [0.1, 0.12, 0.11, 0.13, 0.09]),
        ("policy_low", "policy", "low", [0.01, -0.02, 0.03, -0.01, 0.0]),
    ]:
        rows += [
            {"condition": cond, "mechanism": mech, "intensity": inten, "seed": s, "effect": v}
            for s, v in enumerate(vals)
        ]
    out = summarize_seed_effects(pd.DataFrame(rows), n_boot=500)
    assert set(out["condition"]) == {"dynamics_high", "policy_low"}
    assert (out["p_holm"] >= out["p_signflip"]).all()
    assert out.set_index("condition").loc["dynamics_high", "n_positive_seeds"] == 5
