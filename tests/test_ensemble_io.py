"""Checkpoint round-trip: min/max log-var must survive save/load (configs C/D use -20)."""
import pytest

torch = pytest.importorskip("torch")

from src.model.ensemble import EnsembleWorldModel  # noqa: E402


def test_log_var_bounds_roundtrip(tmp_path):
    m = EnsembleWorldModel(obs_dim=3, action_dim=2, n_members=2, min_log_var=-20.0)
    m.save(tmp_path)
    m2 = EnsembleWorldModel.load(tmp_path)
    assert all(mem.min_log_var == -20.0 for mem in m2.members)
    assert all(mem.max_log_var == 2.0 for mem in m2.members)
