"""Golden test: frozen_metrics.metric_bundle must match the pipeline
implementation bit-for-bit on Hopper seed_00."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minari
import numpy as np
import pytest

from src.calibration.frozen_metrics import metric_bundle
from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.model.ensemble_v2 import load_member, predict_ensemble

PAPER1 = Path.home() / 'paper1_run' / 'experiments' / 'hopper'


@pytest.mark.skipif(not PAPER1.exists(), reason='legacy data not available')
def test_metric_bundle_matches_legacy():
    dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=False)
    split = deterministic_episode_split(
        dataset.total_episodes, seed=1000,
        train_fraction=0.70, cal_fraction=0.10,
        test_fraction=0.10, probe_fraction=0.10,
    )
    test_obs, test_act, test_y = extract_transitions(
        dataset, split['test'], max_transitions=25000, seed=1002)
    legacy = PAPER1 / 'seed_00'
    members = [load_member(legacy / 'checkpoints' / f'ensemble_member_{i:02d}.pt')
               for i in range(5)]
    pred = predict_ensemble(members, test_obs, test_act)
    metrics = metric_bundle(pred, test_y)

    with open(legacy / 'results' / 'baseline_metrics.json', encoding='utf-8') as f:
        target = json.load(f)

    for k in ['calibration_error', 'nll', 'coverage_50', 'coverage_90',
              'sharpness_50', 'sharpness_90', 'crps']:
        assert math.isclose(metrics[k], target[k], rel_tol=1e-12, abs_tol=1e-12), \
            f'{k}: {metrics[k]} vs {target[k]}'


@pytest.mark.skipif(not PAPER1.exists(), reason='legacy data not available')
def test_bootstrap_and_t_ci_deterministic():
    from src.stats import bootstrap_mean_ci, paired_permutation_pvalue, t_ci

    diffs = np.array([0.01, 0.02, -0.005, 0.015, 0.008])
    m1, lo1, hi1 = bootstrap_mean_ci(diffs, n_boot=1000, seed=42)
    m2, lo2, hi2 = bootstrap_mean_ci(diffs, n_boot=1000, seed=42)
    assert m1 == m2 and lo1 == lo2 and hi1 == hi2

    m, lo, hi = t_ci(diffs)
    assert lo < m < hi

    p = paired_permutation_pvalue(diffs, n_perm=1000, seed=42)
    assert 0 <= p <= 1
