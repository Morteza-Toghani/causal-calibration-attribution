"""Ensemble uncertainty aggregation and the common world-model interface.

Only the framework-independent part of the world model lives here: how the
predictive distributions of K Gaussian members are combined, and the interface
a model must expose. The trained PyTorch ensemble itself is not part of this
repository yet (see ``docs/status.md``).

For members ``k = 1..K`` with ``p_k = N(mu_k, sigma_k^2)``::

    mean      = mean_k mu_k
    aleatoric = mean_k sigma_k^2
    epistemic = Var_k mu_k          (population variance over members)
    total     = aleatoric + epistemic   (variance of the equal-weight mixture)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


def aggregate_gaussian_ensemble(mus: np.ndarray, variances: np.ndarray) -> dict:
    """Combine member predictions of shape ``(K, N, D)`` into mixture moments."""
    mus = np.asarray(mus, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if mus.ndim != 3 or mus.shape != variances.shape:
        raise ValueError("mus and variances must both have shape (K, N, D)")
    if np.any(variances <= 0):
        raise ValueError("member variances must be strictly positive")
    aleatoric = variances.mean(axis=0)
    epistemic = mus.var(axis=0)  # ddof=0: variance of the mixture
    return {
        "mean": mus.mean(axis=0),
        "aleatoric_var": aleatoric,
        "epistemic_var": epistemic,
        "total_var": aleatoric + epistemic,
    }


def aleatoric_fraction(agg: dict) -> float:
    """Share of total predictive variance that is aleatoric (pooled over N, D)."""
    return float(agg["aleatoric_var"].sum() / agg["total_var"].sum())


class WorldModel(Protocol):
    """Common interface for probabilistic one-step world models."""

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> None: ...

    def predict_dist(self, states: np.ndarray, actions: np.ndarray) -> dict:
        """Return ``{"mean", "total_var", ...}``, each ``(N, D)``."""
        ...
