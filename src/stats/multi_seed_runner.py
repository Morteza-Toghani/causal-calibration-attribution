"""Aggregate per-seed effects into one auditable summary row per condition."""

from __future__ import annotations

import pandas as pd

from .confidence_intervals import bootstrap_mean_ci, t_mean_ci
from .effect_sizes import paired_cohens_dz, sign_consistency
from .significance_tests import holm_adjust, signflip_permutation_pvalue


def summarize_seed_effects(
    effects: pd.DataFrame,
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """``effects`` has columns condition, mechanism, intensity, seed, effect.

    The inference unit is the seed. Holm correction is applied across all
    conditions in ``effects`` (one family).
    """
    rows = []
    for cond, g in effects.groupby("condition", sort=True):
        v = g.sort_values("seed")["effect"].to_numpy()
        b_lo, b_hi = bootstrap_mean_ci(v, level, n_boot, seed)
        t_lo, t_hi = t_mean_ci(v, level)
        pos, n = sign_consistency(v)
        rows.append(
            {
                "condition": cond,
                "mechanism": g["mechanism"].iloc[0],
                "intensity": g["intensity"].iloc[0],
                "n_seeds": n,
                "mean_effect": float(v.mean()),
                "sd_across_seeds": float(v.std(ddof=1)),
                "boot_ci_low": b_lo,
                "boot_ci_high": b_hi,
                "t_ci_low": t_lo,
                "t_ci_high": t_hi,
                "cohens_dz": paired_cohens_dz(v),
                "n_positive_seeds": pos,
                "p_signflip": signflip_permutation_pvalue(v, seed=seed),
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_signflip"].to_numpy())
    return out
