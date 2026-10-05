"""Episode-level data splits (no transition-level leakage between splits)."""

from __future__ import annotations

import numpy as np

DEFAULT_FRACTIONS = {"train": 0.70, "development": 0.15, "test": 0.10, "probe": 0.05}


def episode_split(
    n_episodes: int,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Randomly assign whole episodes to splits. Returns sorted episode indices.

    Every episode lands in exactly one split; the last split absorbs rounding.
    """
    fractions = fractions or DEFAULT_FRACTIONS
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    if n_episodes < len(fractions):
        raise ValueError("need at least one episode per split")
    perm = np.random.default_rng(seed).permutation(n_episodes)
    names = list(fractions)
    sizes = [int(round(fractions[n] * n_episodes)) for n in names[:-1]]
    sizes.append(n_episodes - sum(sizes))
    if min(sizes) < 1:
        raise ValueError("split sizes must be >= 1 episode")
    out, start = {}, 0
    for name, size in zip(names, sizes):
        out[name] = np.sort(perm[start : start + size])
        start += size
    return out


def transitions_for_episodes(episode_lengths: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    """Indices of all transitions belonging to ``episodes`` (flat, episode-ordered data)."""
    lengths = np.asarray(episode_lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return np.concatenate([np.arange(starts[e], starts[e] + lengths[e]) for e in np.sort(episodes)])
