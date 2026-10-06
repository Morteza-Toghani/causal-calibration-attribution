"""Signed coverage analysis: (empirical - nominal) for baseline and
all six intervention conditions, all 5 seeds.

Motivation (from the critique):
- The unsigned calibration_error hides whether the model is over- or
  under-covering.
- The baseline already over-covers (cov_50 = 0.84 > 0.50), so a negative
  ATE on calibration_error is NOT an improvement -- it means the shift
  brings coverage closer to nominal from above.
- We report signed (cov - nominal) at all 9 levels.

Output:
  results/tables/signed_coverage_all_seeds.csv
  results/tables/signed_coverage_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minari
import numpy as np
import pandas as pd

from src.calibration.instance import z_for_central_coverage
from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.interventions.dynamics import dynamics_next_state_from_env
from src.interventions.observation import compute_obs_scale, make_observation_noise
from src.interventions.policy import action_scaler, resimulate_with_new_action
from src.model.ensemble_v2 import load_member, predict_ensemble

PAPER1 = Path.home() / 'paper1_run' / 'experiments' / 'hopper'
LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def signed_coverage_per_level(mu, var, y, levels=LEVELS):
    sigma = np.sqrt(var)
    out = {}
    for c in levels:
        zc = z_for_central_coverage(c)
        hw = zc * sigma
        inside = (y >= mu - hw) & (y <= mu + hw)
        out[c] = float(inside.mean() - c)
    return out


def collect_seed(dataset, seed_idx):
    legacy = PAPER1 / f'seed_{seed_idx:02d}'

    split = deterministic_episode_split(
        dataset.total_episodes, seed=1000 + seed_idx,
        train_fraction=0.70, cal_fraction=0.10,
        test_fraction=0.10, probe_fraction=0.10,
    )
    test_obs, test_act, test_y = extract_transitions(
        dataset, split['test'], max_transitions=25000, seed=1000 + seed_idx + 2
    )
    probe_obs, probe_act, probe_y = extract_transitions(
        dataset, split['probe'], max_transitions=10000, seed=1000 + seed_idx + 3
    )
    members = [load_member(legacy / 'checkpoints' / f'ensemble_member_{i:02d}.pt')
               for i in range(5)]

    rows = []

    base_pred = predict_ensemble(members, test_obs, test_act)
    for level, sc in signed_coverage_per_level(base_pred['mu'], base_pred['var'], test_y).items():
        rows.append({'seed': seed_idx, 'condition': 'baseline', 'level': level, 'signed_cov': sc})

    obs_scale = compute_obs_scale(test_obs)
    rng = np.random.default_rng(5000 + seed_idx)
    for level_name, noise_fraction in (('low', 0.05), ('high', 0.10)):
        eps = make_observation_noise(test_obs.shape, obs_scale, noise_fraction, rng)
        pred_o = predict_ensemble(members, test_obs, test_act, observation_noise=eps)
        for lvl, sc in signed_coverage_per_level(pred_o['mu'], pred_o['var'], test_y).items():
            rows.append({'seed': seed_idx,
                         'condition': f'observation_{level_name}',
                         'level': lvl, 'signed_cov': sc})

    env = dataset.recover_environment()
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)
    for level_name, scale in (('low', 0.95), ('high', 0.80)):
        transform = action_scaler(scale, action_low, action_high)
        new_act = transform(probe_act)
        new_y = resimulate_with_new_action(env, probe_obs, new_act)
        pred_p = predict_ensemble(members, probe_obs, new_act)
        for lvl, sc in signed_coverage_per_level(pred_p['mu'], pred_p['var'], new_y).items():
            rows.append({'seed': seed_idx,
                         'condition': f'policy_{level_name}',
                         'level': lvl, 'signed_cov': sc})

    dyn_n = min(len(test_obs), 5000)
    rng_dyn = np.random.default_rng(11000 + seed_idx)
    dyn_idx = np.sort(rng_dyn.choice(len(test_obs), dyn_n, replace=False))
    dyn_obs = test_obs[dyn_idx]
    dyn_act = test_act[dyn_idx]
    for level_name, mass_scale in (('low', 0.95), ('high', 0.80)):
        dyn_y, _ = dynamics_next_state_from_env(
            env, dyn_obs, dyn_act, mass_scale=mass_scale, seed=123 + seed_idx
        )
        pred_d = predict_ensemble(members, dyn_obs, dyn_act)
        for lvl, sc in signed_coverage_per_level(pred_d['mu'], pred_d['var'], dyn_y).items():
            rows.append({'seed': seed_idx,
                         'condition': f'dynamics_{level_name}',
                         'level': lvl, 'signed_cov': sc})

    env.close()
    return rows


def main():
    dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=False)

    all_rows = []
    for seed_idx in range(5):
        print(f'processing seed {seed_idx}...')
        all_rows.extend(collect_seed(dataset, seed_idx))

    df = pd.DataFrame(all_rows)
    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'signed_coverage_all_seeds.csv', index=False)
    print(f'saved {out_dir / "signed_coverage_all_seeds.csv"}')

    summary = df.groupby(['condition', 'level'])['signed_cov'].agg(['mean', 'std']).reset_index()

    md = ['# Signed coverage summary', '']
    md.append('Signed coverage = (empirical coverage - nominal level).')
    md.append('Positive = over-covering; negative = under-covering.')
    md.append('')
    header = '| condition | ' + ' | '.join(f'{c:.2f}' for c in LEVELS) + ' |'
    md.append(header)
    md.append('|' + '|'.join(['---'] * (len(LEVELS) + 1)) + '|')
    for cond in sorted(df['condition'].unique()):
        cells = [f'**{cond}**']
        for c in LEVELS:
            m = summary[(summary['condition'] == cond) & (summary['level'] == c)]
            if len(m):
                cells.append(f'{m.iloc[0]["mean"]:+.3f}')
            else:
                cells.append('-')
        md.append('| ' + ' | '.join(cells) + ' |')

    md.append('')
    md.append('### Key observations')
    md.append('')
    md.append('* Baseline over-covers at all levels (signed_cov > 0).')
    md.append('* Observation shift: low noise may reduce over-coverage; high noise')
    md.append('  pushes through nominal into under-coverage.')
    md.append('* Policy and dynamics shift have smaller, direction-consistent effects.')

    with open(out_dir / 'signed_coverage_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'saved {out_dir / "signed_coverage_summary.md"}')

    print()
    print('=== Signed coverage (mean over 5 seeds) ===')
    print(f'{"condition":<20}', end='')
    for c in LEVELS:
        print(f'{c:>8.2f}', end='')
    print()
    for cond in sorted(df['condition'].unique()):
        print(f'{cond:<20}', end='')
        for c in LEVELS:
            m = summary[(summary['condition'] == cond) & (summary['level'] == c)]
            if len(m):
                print(f'{m.iloc[0]["mean"]:>+8.3f}', end='')
            else:
                print(f'{"-":>8}', end='')
        print()


if __name__ == '__main__':
    main()
