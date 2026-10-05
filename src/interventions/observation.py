"""Observation shift: additive Gaussian corruption with reusable base noise.

``o' = o + noise_fraction * reference_scale * eps`` with ``eps ~ N(0, I)``.

Common random numbers: ``eps`` is drawn ONCE (``draw_base_noise``) and reused for
every intensity, so a high-intensity condition is an exact rescaling of the
low-intensity perturbation.

OPEN ITEM: ``reference_scale`` (e.g. per-dimension std of the training
observations) must be checked against the original experiment code before any
number produced by this function is compared with ``results/raw``.
"""

from __future__ import annotations

import numpy as np


def draw_base_noise(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Standard-normal base noise, fully determined by ``seed``."""
    return np.random.default_rng(seed).standard_normal(shape)


def apply_observation_shift(
    obs: np.ndarray,
    base_noise: np.ndarray,
    noise_fraction: float,
    reference_scale: np.ndarray | float,
) -> np.ndarray:
    obs = np.asarray(obs, dtype=float)
    if base_noise.shape != obs.shape:
        raise ValueError(f"base_noise {base_noise.shape} must match obs {obs.shape}")
    if noise_fraction < 0:
        raise ValueError("noise_fraction must be >= 0")
    return obs + noise_fraction * np.asarray(reference_scale, dtype=float) * base_noise
