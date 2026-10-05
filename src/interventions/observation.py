"""Observation intervention — FROZEN pipeline semantics.

Matches Paper1 pipeline lines 1038-1050:

    obs_scale = np.std(test_obs, axis=0).astype(np.float32)
    rng = np.random.default_rng(5000 + seed_idx)
    for level, noise_level in (('low', 0.05), ('high', 0.10)):
        z = rng.standard_normal(size=test_obs.shape).astype(np.float32)
        eps = z * (obs_scale + 1e-6) * float(noise_level)
        # eps added to obs via predict_ensemble(observation_noise=eps)

The intervention is applied to the *input* observation only;
the target y_true is unchanged.
"""
from __future__ import annotations

import numpy as np


def compute_obs_scale(test_obs: np.ndarray) -> np.ndarray:
    """Reference scale = per-dim std of the test set."""
    return np.std(test_obs, axis=0).astype(np.float32)


def make_observation_noise(
    obs_shape: tuple,
    obs_scale: np.ndarray,
    noise_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return eps of shape obs_shape to be added to observations.

    eps ~ Normal(0, (obs_scale + 1e-6) * noise_fraction)
    """
    z = rng.standard_normal(size=obs_shape).astype(np.float32)
    return z * (obs_scale + 1e-6) * float(noise_fraction)


def apply_observation_noise(obs: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Additive intervention on input observations."""
    return (obs + eps).astype(np.float32)
