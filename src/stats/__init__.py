"""Statistical inference for the paper."""
from src.stats.intervals import (
    bootstrap_mean_ci,
    paired_permutation_pvalue,
    t_ci,
)

__all__ = [
    'bootstrap_mean_ci',
    'paired_permutation_pvalue',
    't_ci',
]
