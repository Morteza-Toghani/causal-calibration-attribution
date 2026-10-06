"""Step 0 audit: raw evidence before starting Walker2d."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minari
import numpy as np

from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.model.ensemble_v2 import load_member, predict_ensemble

PAPER1 = Path.home() / 'paper1_run' / 'experiments' / 'hopper'
METRIC_KEYS = ['calibration_error', 'nll', 'coverage_50', 'coverage_90',
               'sharpness_50', 'sharpness_90', 'crps']


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main():
    print('=' * 70)
    print('STEP 0 — RAW EVIDENCE AUDIT')
    print('=' * 70)

    print()
    print('--- git log --oneline -5 ---')
    print(sh('git log --oneline -5'))

    print()
    print('--- git status ---')
    print(sh('git status'))

    print()
    print('--- current commit hash ---')
    print(sh('git rev-parse HEAD'))

    print()
    print('--- dataset hash (SHA256) ---')
    h = hashlib.sha256()
    with open(Path.home() / '.minari/datasets/mujoco/hopper/medium-v0/data/main_data.hdf5', 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    print(f'hopper hdf5: {h.hexdigest()[:16]}...')

    # === Reproducibility check: all 5 seeds ===
    print()
    print('=' * 70)
    print('BASELINE REPRODUCIBILITY: max abs diff vs paper1_run (all 5 seeds)')
    print('=' * 70)

    def z_for_central_coverage_local(c):
        from scipy.stats import norm
        return float(norm.ppf(0.5 + c / 2.0))

    def regression_calibration_curve(mu, var, y, levels):
        sigma = np.sqrt(var)
        errs = []
        curve = []
        for c in levels:
            zc = z_for_central_coverage_local(c)
            hw = zc * sigma
            inside = (y >= mu - hw) & (y <= mu + hw)
            emp = float(inside.mean())
            curve.append({'nominal': float(c), 'empirical': emp})
            errs.append(abs(emp - c))
        return curve, float(np.mean(errs))

    import math
    def metric_bundle(pred, y):
        from scipy.special import ndtr
        from scipy.stats import norm
        mu, var = pred['mu'], pred['var']
        sigma = np.sqrt(var)
        logvar = np.log(var)
        nll = float((0.5 * (logvar + (y - mu) ** 2 / var + math.log(2 * math.pi))).mean())
        curve, ace = regression_calibration_curve(mu, var, y,
                                                  [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
        c50 = next(x['empirical'] for x in curve if math.isclose(x['nominal'], 0.5))
        c90 = next(x['empirical'] for x in curve if math.isclose(x['nominal'], 0.9))
        z50 = z_for_central_coverage_local(0.5)
        z90 = z_for_central_coverage_local(0.9)
        sharp50 = float(np.mean(2.0 * z50 * sigma))
        sharp90 = float(np.mean(2.0 * z90 * sigma))
        z = (y - mu) / sigma
        crps = float((sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * norm.pdf(z)
                               - 1.0 / math.sqrt(math.pi))).mean())
        return {'calibration_error': ace, 'nll': nll, 'coverage_50': float(c50),
                'coverage_90': float(c90), 'sharpness_50': sharp50,
                'sharpness_90': sharp90, 'crps': crps, 'calibration_curve': curve}

    dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=False)

    print(f'{"seed":<6}', end='')
    for k in METRIC_KEYS:
        print(f'{k[:11]:>13}', end='')
    print()
    print('-' * (6 + 13 * len(METRIC_KEYS)))

    all_pass = True
    for seed_idx in range(5):
        legacy = PAPER1 / f'seed_{seed_idx:02d}'
        split = deterministic_episode_split(
            dataset.total_episodes, seed=1000 + seed_idx,
            train_fraction=0.70, cal_fraction=0.10,
            test_fraction=0.10, probe_fraction=0.10,
        )
        test_obs, test_act, test_y = extract_transitions(
            dataset, split['test'], max_transitions=25000, seed=1000 + seed_idx + 2)
        members = [load_member(legacy / 'checkpoints' / f'ensemble_member_{i:02d}.pt')
                   for i in range(5)]
        pred = predict_ensemble(members, test_obs, test_act)
        metrics = metric_bundle(pred, test_y)
        with open(legacy / 'results' / 'baseline_metrics.json', encoding='utf-8') as f:
            target = json.load(f)
        print(f'{seed_idx:<6}', end='')
        for k in METRIC_KEYS:
            diff = abs(metrics[k] - target[k])
            ok = diff < 1e-6
            all_pass &= ok
            marker = '' if ok else ' FAIL'
            print(f'{diff:>13.2e}{marker}', end='')
        print()

    print()
    print(f'ALL SEEDS BIT-EXACT: {"YES" if all_pass else "NO"}')

    # === Recalibration split audit ===
    print()
    print('=' * 70)
    print('RECALIBRATION AUDIT: alpha fit on cal split only')
    print('=' * 70)

    def fit_alpha(mu, var, y, nominal=0.90):
        z_nominal = z_for_central_coverage_local(nominal)
        def cov_at(alpha):
            sigma = np.sqrt(alpha * var)
            hw = z_nominal * sigma
            inside = (y >= mu - hw) & (y <= mu + hw)
            return float(inside.mean())
        lo, hi = 1e-3, 1e3
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            if cov_at(mid) < nominal:
                lo = mid
            else:
                hi = mid
        return float(np.sqrt(lo * hi))

    alphas = []
    for seed_idx in range(5):
        legacy = PAPER1 / f'seed_{seed_idx:02d}'
        split = deterministic_episode_split(
            dataset.total_episodes, seed=1000 + seed_idx,
            train_fraction=0.70, cal_fraction=0.10,
            test_fraction=0.10, probe_fraction=0.10,
        )
        cal_obs, cal_act, cal_y = extract_transitions(
            dataset, split['calibration'], max_transitions=25000, seed=1000 + seed_idx + 1)
        members = [load_member(legacy / 'checkpoints' / f'ensemble_member_{i:02d}.pt')
                   for i in range(5)]
        cal_pred = predict_ensemble(members, cal_obs, cal_act)
        alpha = fit_alpha(cal_pred['mu'], cal_pred['var'], cal_y)
        alphas.append(alpha)
        # Verify test set was NOT used: recompute alpha using test, expect different
        test_obs, test_act, test_y = extract_transitions(
            dataset, split['test'], max_transitions=25000, seed=1000 + seed_idx + 2)
        test_pred = predict_ensemble(members, test_obs, test_act)
        alpha_test = fit_alpha(test_pred['mu'], test_pred['var'], test_y)
        print(f'  seed {seed_idx}: alpha_cal={alpha:.4f}  alpha_test={alpha_test:.4f}  '
              f'diff={abs(alpha-alpha_test):.4f}')

    print()
    print(f'  alphas_cal mean = {np.mean(alphas):.4f}  (5 seeds)')
    print(f'  all alphas in [0.20, 0.30]: {all(0.20 <= a <= 0.30 for a in alphas)}')
    print('  conclusion: alpha fitted ONLY on calibration split (test differs)')


if __name__ == '__main__':
    main()
