"""Effect sizes for paired / per-seed effects."""

from __future__ import annotations

import numpy as np


def paired_cohens_dz(effects) -> float:
    """Cohen's d_z = mean(d) / sd(d, ddof=1) for paired differences ``d``.

    Returns ``nan`` when sd == 0 or fewer than 2 values (undefined).
    """
    d = np.asarray(effects, float)
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float("nan") if sd == 0 else float(d.mean() / sd)


def sign_consistency(effects) -> tuple[int, int]:
    """(# strictly positive effects, total)."""
    d = np.asarray(effects, float)
    return int((d > 0).sum()), int(d.size)
