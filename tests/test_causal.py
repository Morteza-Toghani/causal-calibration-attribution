import numpy as np
import pandas as pd
import pytest

from src.causal.attribution import (
    assert_matched,
    average_treatment_effect,
    paired_differences,
    seed_level_effects,
    split_condition,
)


def test_ate_is_mean_of_paired_differences():
    b, t = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.5, 4.5])
    assert average_treatment_effect(b, t) == pytest.approx(1.0)
    assert paired_differences(b, t).tolist() == [1.0, 0.5, 1.5]


def test_null_intervention_has_zero_ate():
    b = np.random.default_rng(0).normal(size=100)
    assert average_treatment_effect(b, b.copy()) == 0.0


def test_shape_mismatch_and_unmatched_instances_raise():
    with pytest.raises(ValueError):
        paired_differences([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        assert_matched([0, 1, 2], [0, 2, 1])
    assert_matched([0, 1, 2], [0, 1, 2])


def test_split_condition():
    assert split_condition("dynamics_high") == ("dynamics", "high")
    assert split_condition("baseline") == ("baseline", "")


def test_seed_level_effects_matches_manual_difference():
    df = pd.DataFrame(
        {
            "seed_idx": [0, 0, 1, 1],
            "condition": ["baseline", "policy_high", "baseline", "policy_high"],
            "m": [0.2, 0.25, 0.3, 0.29],
        }
    )
    out = seed_level_effects(df, "m").sort_values("seed")
    assert out["effect"].tolist() == pytest.approx([0.05, -0.01])
    assert set(out["mechanism"]) == {"policy"}


def test_seed_level_effects_requires_baseline_per_seed():
    df = pd.DataFrame({"seed_idx": [0], "condition": ["policy_high"], "m": [1.0]})
    with pytest.raises(ValueError):
        seed_level_effects(df, "m")
