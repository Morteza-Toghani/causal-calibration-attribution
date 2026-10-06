"""Golden test: verify loading legacy checkpoint works."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.model.ensemble_v2 import load_member, predict_ensemble

LEGACY = Path.home() / "paper1_run" / "experiments" / "hopper" / "seed_00"


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy data not available")
def test_load_one_member():
    ckpt = LEGACY / "checkpoints" / "ensemble_member_00.pt"
    member = load_member(ckpt)
    assert member.model is not None
    assert len(member.standardizer.mean_x) == 14
    assert len(member.standardizer.mean_y) == 11


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy data not available")
def test_predict_shape():
    ckpt = LEGACY / "checkpoints" / "ensemble_member_00.pt"
    member = load_member(ckpt)
    obs = np.zeros((10, 11), dtype=np.float32)
    act = np.zeros((10, 3), dtype=np.float32)
    pred = predict_ensemble([member], obs, act)
    assert pred["mu"].shape == (10, 11)
    assert pred["var"].shape == (10, 11)
    assert (pred["var"] > 0).all()


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy data not available")
def test_load_all_5_members():
    members = [load_member(LEGACY / "checkpoints" / f"ensemble_member_{i:02d}.pt") for i in range(5)]
    assert len(members) == 5
