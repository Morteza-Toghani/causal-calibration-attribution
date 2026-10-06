"""Reproduce baseline_metrics.json for Hopper seed_00.

Target (from paper1_run/.../seed_00/results/baseline_metrics.json):
  calibration_error = 0.23981979797979797
  nll              = -2.7825677394866943
  coverage_50      = 0.8429854545454546
  coverage_90      = 0.9816254545454545
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.model.ensemble_v2 import load_member, predict_ensemble

LEGACY = Path.home() / 'paper1_run' / 'experiments' / 'hopper' / 'seed_00'
TARGET_METRICS = LEGACY / 'results' / 'baseline_metrics.json'

NOMINAL_COVERAGES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def z_for_central_coverage(c):
    from scipy.stats import norm
    return float(norm.ppf(0.5 + c / 2.0))


def regression_calibration_curve(mu, var, y, levels=NOMINAL_COVERAGES):
    sigma = np.sqrt(var)
    curve = []
    errors = []
    for c in levels:
        z = z_for_central_coverage(c)
        hw = z * sigma
        inside = (y >= mu - hw) & (y <= mu + hw)
        emp = float(inside.mean())
        curve.append({'nominal': float(c), 'empirical': emp})
        errors.append(abs(emp - c))
    ace = float(np.mean(errors))
    return curve, ace


def metric_bundle(pred, y):
    from scipy.special import ndtr
    from scipy.stats import norm

    mu = pred['mu']
    var = pred['var']
    sigma = np.sqrt(var)

    logvar = np.log(var)
    nll = 0.5 * (logvar + (y - mu) ** 2 / var + math.log(2.0 * math.pi))
    nll = float(nll.mean())

    curve, ace = regression_calibration_curve(mu, var, y)

    c50 = next(x['empirical'] for x in curve if math.isclose(x['nominal'], 0.5))
    c90 = next(x['empirical'] for x in curve if math.isclose(x['nominal'], 0.9))

    z50 = z_for_central_coverage(0.5)
    z90 = z_for_central_coverage(0.9)
    sharp50 = float(np.mean(2.0 * z50 * sigma))
    sharp90 = float(np.mean(2.0 * z90 * sigma))

    z = (y - mu) / sigma
    crps = sigma * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )
    crps = float(np.mean(crps))

    return {
        'calibration_error': ace,
        'nll': nll,
        'coverage_50': float(c50),
        'coverage_90': float(c90),
        'sharpness_50': sharp50,
        'sharpness_90': sharp90,
        'crps': crps,
        'calibration_curve': curve,
    }


def main():
    import minari

    print('=== Reproduce baseline for Hopper seed_00 ===')

    # 1. Load dataset from cache
    dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=False)
    print(f'dataset episodes: {dataset.total_episodes}')
    print(f'dataset steps:    {dataset.total_steps}')

    # 2. Reproduce FROZEN split with seed=1000 (1000 + seed_idx=0)
    split = deterministic_episode_split(
        dataset.total_episodes,
        seed=1000,
        train_fraction=0.70,
        cal_fraction=0.10,
        test_fraction=0.10,
        probe_fraction=0.10,
    )
    print(f'split train: {len(split["train"])}, cal: {len(split["calibration"])}, '
          f'test: {len(split["test"])}, probe: {len(split["probe"])}')

    # Verify against paper1_run manifest
    with open(LEGACY / 'split_manifest.json', encoding='utf-8') as f:
        legacy_split = json.load(f)
    assert split['train'] == legacy_split['train_episode_ids'], 'train mismatch'
    assert split['test'] == legacy_split['test_episode_ids'], 'test mismatch'
    assert split['calibration'] == legacy_split['calibration_episode_ids'], 'cal mismatch'
    assert split['probe'] == legacy_split['probe_episode_ids'], 'probe mismatch'
    print('✓ split matches legacy manifest')

    # 3. Extract test transitions (seed = 1000 + 2 = 1002, max = 25000)
    test_obs, test_act, test_y = extract_transitions(
        dataset, split['test'], max_transitions=25000, seed=1002
    )
    print(f'test transitions: {len(test_obs)}')

    # 4. Load 5 checkpoints
    members = [
        load_member(LEGACY / 'checkpoints' / f'ensemble_member_{i:02d}.pt')
        for i in range(5)
    ]
    print(f'loaded {len(members)} members')

    # 5. Predict
    pred = predict_ensemble(members, test_obs, test_act)
    print(f'pred mu shape: {pred["mu"].shape}, var shape: {pred["var"].shape}')

    # 6. Compute metrics
    metrics = metric_bundle(pred, test_y)

    # 7. Compare with baseline_metrics.json
    with open(TARGET_METRICS, encoding='utf-8') as f:
        target = json.load(f)

    print()
    print('=== Comparison ===')
    print(f'{"metric":<20} {"reproduced":>15} {"target":>15} {"rel_err":>12}')
    print('-' * 65)
    for k in ['calibration_error', 'nll', 'coverage_50', 'coverage_90',
              'sharpness_50', 'sharpness_90', 'crps']:
        rep = metrics[k]
        tgt = target[k]
        rel = abs(rep - tgt) / (abs(tgt) + 1e-12)
        marker = '✓' if rel < 1e-4 else '✗'
        print(f'{k:<20} {rep:>15.10f} {tgt:>15.10f} {rel:>12.2e} {marker}')

    # Save
    out_dir = Path('results/reproduction')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'baseline_seed00.json', 'w', encoding='utf-8') as f:
        json.dump({'reproduced': metrics, 'target': target}, f, indent=2)
    print()
    print(f'saved: {out_dir / "baseline_seed00.json"}')


if __name__ == '__main__':
    main()
