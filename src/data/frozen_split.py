"""FROZEN pipeline split logic — exact copy from Paper1 pipeline."""
from __future__ import annotations

import math

import numpy as np


def deterministic_episode_split(
    total_episodes: int,
    seed: int,
    train_fraction: float,
    cal_fraction: float,
    test_fraction: float,
    probe_fraction: float,
):
    total = train_fraction + cal_fraction + test_fraction + probe_fraction
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f'Fractions must sum to 1.0, got {total}')

    rng = np.random.default_rng(seed)
    idx = np.arange(total_episodes)
    rng.shuffle(idx)

    n_train = int(round(total_episodes * train_fraction))
    n_cal = int(round(total_episodes * cal_fraction))
    n_test = int(round(total_episodes * test_fraction))
    n_probe = total_episodes - n_train - n_cal - n_test

    p0 = 0
    p1 = p0 + n_train
    p2 = p1 + n_cal
    p3 = p2 + n_test

    return {
        'train': idx[p0:p1].tolist(),
        'calibration': idx[p1:p2].tolist(),
        'test': idx[p2:p3].tolist(),
        'probe': idx[p3:p3 + n_probe].tolist(),
    }
