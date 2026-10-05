"""Input/output standardizer — matches Paper 1 FROZEN pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    mean_x: list
    std_x: list
    mean_y: list
    std_y: list


def fit_standardizer(obs: np.ndarray, act: np.ndarray, nxt: np.ndarray) -> Standardizer:
    x = np.concatenate([obs, act], axis=1)
    mx = x.mean(axis=0)
    sx = np.maximum(x.std(axis=0), 1e-6)
    my = nxt.mean(axis=0)
    sy = np.maximum(nxt.std(axis=0), 1e-6)
    return Standardizer(mx.tolist(), sx.tolist(), my.tolist(), sy.tolist())


def apply_standardizer(standardizer, obs, act, nxt=None):
    mx = np.asarray(standardizer.mean_x, dtype=np.float32)
    sx = np.asarray(standardizer.std_x, dtype=np.float32)
    my = np.asarray(standardizer.mean_y, dtype=np.float32)
    sy = np.asarray(standardizer.std_y, dtype=np.float32)
    x = np.concatenate([obs, act], axis=1).astype(np.float32)
    xz = (x - mx) / sx
    if nxt is None:
        return xz
    yz = (nxt.astype(np.float32) - my) / sy
    return xz, yz
