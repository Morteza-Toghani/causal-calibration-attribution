"""Reproduce baseline_metrics.json for all 5 Hopper seeds."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minari
import numpy as np

from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.model.ensemble_v2 import load_member, predict_ensemble

PAPER1 = Path.home() / 'paper1_run' / 'experiments' / 'hopper'
NOMINAL_COVERAGES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
METRIC_KEYS = ['calibration_error', 'nll', 'coverage_50', 'coverage_90',
               'sharpness_50', 'sharpness_90', 'crps']


def z_for_central_coverage(c):
    from scipy.stats import norm
    return float(norm.ppf(0.5 + c / 2.0))


def regression_calibration_curve(mu, var, y, levels=NOMINAL_COVERAGES):
    sigma = np.sqrt(var)
    curve, errors = [], []
    for c in levels:
        z = z_for_central_coverage(c)
        hw = z * sigma
        inside = (y >= mu - hw) & (y <= mu + hw)
        emp = float(inside.mean())
        curve.append({'nominal': float(c), 'empirical': emp})
        errors.append(abs(emp - c))
    return curve, float(np.mean(errors))


def metric_bundle(pred, y):
    from scipy.special import ndtr
    from scipy.stats import norm

    mu, var = pred['mu'], pred['var']
    sigma = np.sqrt(var)
    logvar = np.log(var)
    nll = float((0.5 * (logvar + (y - mu) ** 2 / var + math.log(2.0 * math.pi))).mean())

    curve, ace = regression_calibration_curve(mu, var, y)
    c50 = next(x['empirical'] for x in curve if math.isclose(x['nominal'], 0.5))
    c90 = next(x['empirical'] for x in curve if math.isclose(x['nominal'], 0.9))
    z50 = z_for_central_coverage(0.5)
    z90 = z_for_central_coverage(0.9)
    sharp50 = float(np.mean(2.0 * z50 * sigma))
    sharp90 = float(np.mean(2.0 * z90 * sigma))

    z = (y - mu) / sigma
    crps = float((sigma * (
        z * (2.0 * ndtr(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi)
    )).mean())

    return {
        'calibration_error': ace, 'nll': nll,
        'coverage_50': float(c50), 'coverage_90': float(c90),
        'sharpness_50': sharp50, 'sharpness_90': sharp90, 'crps': crps,
        'calibration_curve': curve,
    }


def reproduce_seed(dataset, seed_idx):
    legacy = PAPER1 / f'seed_{seed_idx:02d}'
    split = deterministic_episode_split(
        dataset.total_episodes, seed=1000 + seed_idx,
        train_fraction=0.70, cal_fraction=0.10,
        test_fraction=0.10, probe_fraction=0.10,
    )

    # verify against manifest
    with open(legacy / 'split_manifest.json', encoding='utf-8') as f:
        legacy_split = json.load(f)
    for k, lk in [('train', 'train_episode_ids'), ('calibration', 'calibration_episode_ids'),
                  ('test', 'test_episode_ids'), ('probe', 'probe_episode_ids')]:
        assert split[k] == legacy_split[lk], f'{k} mismatch in seed {seed_idx}'

    # extract test (seed = 1000 + seed_idx + 2)
    test_obs, test_act, test_y = extract_transitions(
        dataset, split['test'], max_transitions=25000, seed=1000 + seed_idx + 2
    )

    # load 5 members
    members = [load_member(legacy / 'checkpoints' / f'ensemble_member_{i:02d}.pt')
               for i in range(5)]

    pred = predict_ensemble(members, test_obs, test_act)
    metrics = metric_bundle(pred, test_y)

    with open(legacy / 'results' / 'baseline_metrics.json', encoding='utf-8') as f:
        target = json.load(f)

    return metrics, target


def main():
    dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=False)

    print(f'{"seed":<6} ' + ' '.join(f'{k[:9]:>11}' for k in METRIC_KEYS))
    print('-' * (6 + 12 * len(METRIC_KEYS)))
    all_ok = True
    for seed_idx in range(5):
        metrics, target = reproduce_seed(dataset, seed_idx)
        row = [f'{seed_idx:02d}']
        for k in METRIC_KEYS:
            rel = abs(metrics[k] - target[k]) / (abs(target[k]) + 1e-12)
            ok = rel < 1e-6
            all_ok &= ok
            row.append(f'{metrics[k]:>11.6f}' + ('' if ok else ' ✗'))
        print('     '.join(row))

    print()
    print('ALL SEEDS PASS' if all_ok else 'SOME SEEDS FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
