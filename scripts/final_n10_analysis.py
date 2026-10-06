"""Final n=10 analysis across Hopper and Walker2d.

Combines:
  - Hopper seeds 0..4 (legacy checkpoints in paper1_run)
  - Hopper seeds 5..9 (new checkpoints in results/ensemble_training)
  - Walker2d seeds 0..4 (new)

Runs the full pipeline (corrected ATE + recalibrated ATE + t-CI) and
writes final CSVs and Markdown tables for the manuscript.

Usage:
    python scripts/final_n10_analysis.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import minari
from scipy.stats import t as t_dist

from src.calibration.frozen_metrics import z_for_central_coverage
from src.calibration.instance import instance_calibration_error
from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.interventions.dynamics import dynamics_next_state_from_env
from src.interventions.observation import compute_obs_scale, make_observation_noise
from src.interventions.policy import action_scaler, resimulate_with_new_action
from src.model.ensemble_v2 import load_member, predict_ensemble
from src.stats import bootstrap_mean_ci, paired_permutation_pvalue, t_ci


DATASET_IDS = {
    'hopper': 'mujoco/hopper/medium-v0',
    'walker2d': 'mujoco/walker2d/medium-v0',
}
LEGACY_HOPPER = Path.home() / 'paper1_run' / 'experiments' / 'hopper'
NEW_ROOT = Path('results/ensemble_training').resolve()
NOMINAL_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
NOMINAL_RECAL = 0.90


def ckpt_root(env_name, seed_idx):
    if env_name == 'hopper' and seed_idx in range(5):
        return LEGACY_HOPPER / f'seed_{seed_idx:02d}' / 'checkpoints'
    return NEW_ROOT / env_name / f'seed_{seed_idx:02d}' / 'checkpoints'


def fit_alpha(mu, var, y, nominal=NOMINAL_RECAL):
    z_nominal = z_for_central_coverage(nominal)
    def cov_at(alpha):
        sigma = np.sqrt(alpha * var)
        hw = z_nominal * sigma
        return float(((y >= mu - hw) & (y <= mu + hw)).mean())
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if cov_at(mid) < nominal:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def process_seed(env_name, dataset, seed_idx):
    root = ckpt_root(env_name, seed_idx)
    members = [load_member(root / f'ensemble_member_{i:02d}.pt') for i in range(5)]

    split = deterministic_episode_split(
        dataset.total_episodes, seed=1000 + seed_idx,
        train_fraction=0.70, cal_fraction=0.10,
        test_fraction=0.10, probe_fraction=0.10,
    )
    cal_obs, cal_act, cal_y = extract_transitions(
        dataset, split['calibration'], max_transitions=25000, seed=1000 + seed_idx + 1)
    test_obs, test_act, test_y = extract_transitions(
        dataset, split['test'], max_transitions=25000, seed=1000 + seed_idx + 2)
    probe_obs, probe_act, probe_y = extract_transitions(
        dataset, split['probe'], max_transitions=10000, seed=1000 + seed_idx + 3)

    cal_pred = predict_ensemble(members, cal_obs, cal_act)
    alpha = fit_alpha(cal_pred['mu'], cal_pred['var'], cal_y)

    env = dataset.recover_environment()
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    # Baselines
    base_pred = predict_ensemble(members, test_obs, test_act)
    base_inst = instance_calibration_error(base_pred['mu'], base_pred['var'], test_y)

    dyn_n = min(len(test_obs), 5000)
    rng_dyn = np.random.default_rng(11000 + seed_idx)
    dyn_idx = np.sort(rng_dyn.choice(len(test_obs), dyn_n, replace=False))
    dyn_obs, dyn_act, dyn_y = test_obs[dyn_idx], test_act[dyn_idx], test_y[dyn_idx]
    pred_dyn = predict_ensemble(members, dyn_obs, dyn_act)

    y_dyn_resim_1, _ = dynamics_next_state_from_env(
        env, dyn_obs, dyn_act, mass_scale=1.0, seed=123 + seed_idx)
    base_dyn_inst = instance_calibration_error(pred_dyn['mu'], pred_dyn['var'], y_dyn_resim_1)

    base_probe_pred = predict_ensemble(members, probe_obs, probe_act)
    y_probe_resim_1 = resimulate_with_new_action(env, probe_obs, probe_act)
    base_probe_inst = instance_calibration_error(
        base_probe_pred['mu'], base_probe_pred['var'], y_probe_resim_1)

    rows = []

    # Observation
    obs_scale = compute_obs_scale(test_obs)
    rng_obs = np.random.default_rng(5000 + seed_idx)
    for level, nf in (('low', 0.05), ('high', 0.10)):
        eps = make_observation_noise(test_obs.shape, obs_scale, nf, rng_obs)
        pred_o = predict_ensemble(members, test_obs, test_act, observation_noise=eps)
        treat = instance_calibration_error(pred_o['mu'], pred_o['var'], test_y)
        diffs = treat - base_inst
        ate, lo, hi = bootstrap_mean_ci(diffs, seed=7000 + seed_idx)
        p = paired_permutation_pvalue(diffs, n_perm=5000, seed=8000 + seed_idx)
        rows.append({'env': env_name, 'seed': seed_idx, 'mechanism': 'observation',
                     'intensity': level, 'ate': ate, 'ci_low': lo, 'ci_high': hi,
                     'p_value': p, 'alpha': alpha})

    # Policy (corrected)
    for level, scale in (('low', 0.95), ('high', 0.80)):
        tf = action_scaler(scale, action_low, action_high)
        new_act = tf(probe_act)
        y_new = resimulate_with_new_action(env, probe_obs, new_act)
        pred_p = predict_ensemble(members, probe_obs, new_act)
        treat = instance_calibration_error(pred_p['mu'], pred_p['var'], y_new)
        diffs = treat - base_probe_inst
        ate, lo, hi = bootstrap_mean_ci(diffs, seed=9000 + seed_idx)
        p = paired_permutation_pvalue(diffs, n_perm=5000, seed=10000 + seed_idx)
        rows.append({'env': env_name, 'seed': seed_idx, 'mechanism': 'policy',
                     'intensity': level, 'ate': ate, 'ci_low': lo, 'ci_high': hi,
                     'p_value': p, 'alpha': alpha})

    # Dynamics (corrected)
    for level, ms in (('low', 0.95), ('high', 0.80)):
        y_dyn, _ = dynamics_next_state_from_env(
            env, dyn_obs, dyn_act, mass_scale=ms, seed=123 + seed_idx)
        treat = instance_calibration_error(pred_dyn['mu'], pred_dyn['var'], y_dyn)
        diffs = treat - base_dyn_inst
        ate, lo, hi = bootstrap_mean_ci(diffs, n_boot=2000, seed=12000 + seed_idx)
        p = paired_permutation_pvalue(diffs, n_perm=3000, seed=13000 + seed_idx)
        rows.append({'env': env_name, 'seed': seed_idx, 'mechanism': 'dynamics',
                     'intensity': level, 'ate': ate, 'ci_low': lo, 'ci_high': hi,
                     'p_value': p, 'alpha': alpha})

    env.close()
    return rows


def main():
    all_rows = []
    for env_name, seeds in (('hopper', range(10)), ('walker2d', range(10))):
        print(f'=== {env_name} ===')
        dataset = minari.load_dataset(DATASET_IDS[env_name], download=False)
        for seed_idx in seeds:
            print(f'  seed {seed_idx}...', end=' ', flush=True)
            t0 = time.time()
            all_rows.extend(process_seed(env_name, dataset, seed_idx))
            print(f'{time.time()-t0:.1f}s')

    df = pd.DataFrame(all_rows)
    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'final_n10_per_seed.csv', index=False)

    print()
    print('=' * 78)
    print('FINAL n=10 (Hopper) / n=5 (Walker2d) — CORRECTED ATE')
    print('=' * 78)

    final_rows = []
    for env_name in ('hopper', 'walker2d'):
        sub = df[df['env'] == env_name]
        print(f'\n--- {env_name} ---')
        print(f'{"condition":<22} {"n":>3} {"mean":>12} {"sd":>10} {"t-CI 95%":>28}')
        print('-' * 78)
        for mech in ('observation', 'dynamics', 'policy'):
            for inten in ('low', 'high'):
                g = sub[(sub['mechanism'] == mech) & (sub['intensity'] == inten)]
                n = len(g)
                m, lo, hi = t_ci(g['ate'].values)
                ci = f'[{lo:+.5f}, {hi:+.5f}]'
                print(f'{mech+"_"+inten:<22} {n:>3} {m:>+12.5f} '
                      f'{g["ate"].std():>10.5f} {ci:>28}')
                final_rows.append({
                    'env': env_name, 'mechanism': mech, 'intensity': inten,
                    'n': n, 'ate_mean': m, 'ate_sd': float(g['ate'].std()),
                    't_ci_low': lo, 't_ci_high': hi,
                    'excludes_zero': (lo > 0) or (hi < 0),
                })

    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(out_dir / 'final_n10_summary.csv', index=False)
    print(f'\nsaved: {out_dir}/final_n10_per_seed.csv')
    print(f'saved: {out_dir}/final_n10_summary.csv')


if __name__ == '__main__':
    main()