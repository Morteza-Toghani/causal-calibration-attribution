"""Paired (matched) counterfactual comparisons and ATE estimation.

Unit: a matched model-evaluation instance (same trained model, same seed, same
evaluation data / probe states, same reusable randomness). The factual outcome
Y_i(0) is the baseline; the outcome of the same instance under mechanism ``m`` is
Y_i(m). ``ATEhat_m = mean_i [Y_i(m) - Y_i(0)]``.

This is a controlled-simulation estimand: the intervention is applied by the
experimenter, so no observational identification argument is needed, but the
comparison is only as clean as the intervention is mechanism-specific
(see docs/causal_assumptions.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assert_matched(baseline_ids, treatment_ids) -> None:
    """Fail loudly if the baseline and treatment instances are not one-to-one."""
    b, t = list(baseline_ids), list(treatment_ids)
    if len(b) != len(t) or b != t:
        raise ValueError("baseline and treatment instances are not matched pair-by-pair")


def paired_differences(baseline, treatment) -> np.ndarray:
    baseline, treatment = np.asarray(baseline, float), np.asarray(treatment, float)
    if baseline.shape != treatment.shape:
        raise ValueError("baseline and treatment must have the same shape")
    return treatment - baseline


def average_treatment_effect(baseline, treatment) -> float:
    return float(paired_differences(baseline, treatment).mean())


def split_condition(condition: str) -> tuple[str, str]:
    """``'dynamics_high' -> ('dynamics', 'high')``; ``'baseline' -> ('baseline', '')``."""
    mech, _, intensity = condition.partition("_")
    return mech, intensity


def seed_level_effects(
    df: pd.DataFrame,
    metric: str,
    seed_col: str = "seed_idx",
    condition_col: str = "condition",
    baseline_name: str = "baseline",
) -> pd.DataFrame:
    """Per-seed ``metric(condition) - metric(baseline)`` for every non-baseline condition.

    Returns columns: seed, condition, mechanism, intensity, effect.
    """
    base = df[df[condition_col] == baseline_name].set_index(seed_col)[metric]
    if base.index.duplicated().any():
        raise ValueError("more than one baseline row per seed")
    rows = []
    for _, r in df[df[condition_col] != baseline_name].iterrows():
        if r[seed_col] not in base.index:
            raise ValueError(f"seed {r[seed_col]} has no baseline row")
        mech, inten = split_condition(r[condition_col])
        rows.append(
            {
                "seed": r[seed_col],
                "condition": r[condition_col],
                "mechanism": mech,
                "intensity": inten,
                "effect": float(r[metric] - base.loc[r[seed_col]]),
            }
        )
    return pd.DataFrame(rows)
