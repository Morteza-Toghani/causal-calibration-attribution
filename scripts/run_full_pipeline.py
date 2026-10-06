"""Unified interventions pipeline (corrected + recalibrated).

Runs, for a given env (hopper or walker2d) and its trained ensemble:
  1. Null control (mass_scale=1.0, action_scale=1.0) -> should be ≈ 0
  2. Corrected ATE (uses y_resim(1.0) as paired baseline for D and P)
  3. Signed coverage (per level, per condition)
  4. t-CI on corrected ATE (n=5 seeds)
  5. Recalibration: per-seed alpha fitted on cal split (nominal 0.90)
  6. Corrected + recalibrated ATE and t-CI

Usage:
    python scripts/run_full_pipeline.py --env hopper
    python scripts/run_full_pipeline.py --env walker2d

For hopper, checkpoints are read from paper1_run/... (legacy).
For walker2d, checkpoints are read from results/ensemble_training/walker2d/.
"""
from __future__ import annotations

import argparse
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

from src.calibration.frozen_metrics import metric_bundle, z_for_central_coverage
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

CKPT_ROOTS = {
    'hopper': Path.home() / 'paper1_run' / 'experiments' / 'hopper',
    'walker2d': Path('results/ensemble_training/walker2d').resolve(),
}

SEEDS = range(5)
NOMINAL_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
NOMINAL_RECAL = 0.90


def fit_alpha(mu, var, y, nominal=NOMINAL_RECAL):
    z_nominal = z_for_central_coverage(nominal)
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


def signed_coverage(mu, var, y, levels=NOMINAL_LEVELS):
    sigma = np.sqrt(var)
    out = {}
    for c in levels:
        zc = z_for_central_coverage(c)
        hw = zc * sigma
        inside = (y >= mu - hw) & (y <= mu + hw)
        out[c] = float(inside.mean() - c)
    return out


def process_seed(env_name, dataset, seed_idx):
    root = CKPT_ROOTS[env_name] / f'seed_{seed_idx:02d}' / 'checkpoints'
    if not root.exists():
        root = CKPT_ROOTS[env_name] / f'seed_{seed_idx:02d}' / 'checkpoints'
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

    # Calibrate alpha on cal split
    cal_pred = predict_ensemble(members, cal_obs, cal_act)
    alpha = fit_alpha(cal_pred['mu'], cal_pred['var'], cal_y)

    env = dataset.recover_environment()
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    rows_null = []
    rows_ate = []
    rows_signed = []

    # === Baseline: y_dataset vs y_resim(1.0) ===
    base_pred = predict_ensemble(members, test_obs, test_act)
    base_inst_dataset = instance_calibration_error(base_pred['mu'], base_pred['var'], test_y)

    # === NULL CONTROL: D (mass=1.0) ===
    dyn_n = min(len(test_obs), 5000)
    rng_dyn = np.random.default_rng(11000 + seed_idx)
    dyn_idx = np.sort(rng_dyn.choice(len(test_obs), dyn_n, replace=False))
    dyn_obs = test_obs[dyn_idx]
    dyn_act = test_act[dyn_idx]
    dyn_base_y = test_y[dyn_idx]

    pred_dyn = predict_ensemble(members, dyn_obs, dyn_act)
    base_dyn_dataset = instance_calibration_error(pred_dyn['mu'], pred_dyn['var'], dyn_base_y)
    y_dyn_resim_1, _ = dynamics_next_state_from_env(
        env, dyn_obs, dyn_act, mass_scale=1.0, seed=123 + seed_idx)
    base_dyn_resim = instance_calibration_error(pred_dyn['mu'], pred_dyn['var'], y_dyn_resim_1)
    null_ate_d = float((base_dyn_resim - base_dyn_dataset).mean())

    # === NULL CONTROL: P (action_scale=1.0) ===
    base_probe_pred = predict_ensemble(members, probe_obs, probe_act)
    base_probe_dataset = instance_calibration_error(base_probe_pred['mu'], base_probe_pred['var'], probe_y)
    y_probe_resim_1 = resimulate_with_new_action(env, probe_obs, probe_act)
    base_probe_resim = instance_calibration_error(base_probe_pred['mu'], base_probe_pred['var'], y_probe_resim_1)
    null_ate_p = float((base_probe_resim - base_probe_dataset).mean())

    rows_null.append({
        'seed': seed_idx, 'null_dynamics_ate': null_ate_d, 'null_policy_ate': null_ate_p,
    })

    # === OBSERVATION (no resim; same baseline as Hopper) ===
    obs_scale = compute_obs_scale(test_obs)
    rng_obs = np.random.default_rng(5000 + seed_idx)
    for level, nf in (('low', 0.05), ('high', 0.10)):
        eps = make_observation_noise(test_obs.shape, obs_scale, nf, rng_obs)
        pred_o = predict_ensemble(members, test_obs, test_act, observation_noise=eps)
        treat = instance_calibration_error(pred_o['mu'], pred_o['var'], test_y)
        diffs = treat - base_inst_dataset
        ate, lo, hi = bootstrap_mean_ci(diffs, seed=7000 + seed_idx)
        p = paired_permutation_pvalue(diffs, n_perm=5000, seed=8000 + seed_idx)
        rows_ate.append({
            'seed': seed_idx, 'mechanism': 'observation', 'intensity': level,
            'ate': ate, 'ci_low': lo, 'ci_high': hi, 'p_value': p,
            'alpha': alpha, 'n_pairs': len(diffs),
        })
        # signed coverage
        for c, sc in signed_coverage(pred_o['mu'], pred_o['var'], test_y).items():
            rows_signed.append({
                'seed': seed_idx, 'mechanism': 'observation', 'intensity': level,
                'level': c, 'signed_cov': sc})

    # === POLICY (corrected: use y_resim(1.0) as baseline for both) ===
    for level, scale in (('low', 0.95), ('high', 0.80)):
        tf = action_scaler(scale, action_low, action_high)
        new_act = tf(probe_act)
        y_new = resimulate_with_new_action(env, probe_obs, new_act)
        pred_p = predict_ensemble(members, probe_obs, new_act)
        treat = instance_calibration_error(pred_p['mu'], pred_p['var'], y_new)
        diffs = treat - base_probe_resim
        ate, lo, hi = bootstrap_mean_ci(diffs, seed=9000 + seed_idx)
        p = paired_permutation_pvalue(diffs, n_perm=5000, seed=10000 + seed_idx)
        rows_ate.append({
            'seed': seed_idx, 'mechanism': 'policy', 'intensity': level,
            'ate': ate, 'ci_low': lo, 'ci_high': hi, 'p_value': p,
            'alpha': alpha, 'n_pairs': len(diffs),
        })
        for c, sc in signed_coverage(pred_p['mu'], pred_p['var'], y_new).items():
            rows_signed.append({
                'seed': seed_idx, 'mechanism': 'policy', 'intensity': level,
                'level': c, 'signed_cov': sc})

    # === DYNAMICS (corrected: use y_resim(1.0) as baseline) ===
    for level, ms in (('low', 0.95), ('high', 0.80)):
        y_dyn, _ = dynamics_next_state_from_env(
            env, dyn_obs, dyn_act, mass_scale=ms, seed=123 + seed_idx)
        treat = instance_calibration_error(pred_dyn['mu'], pred_dyn['var'], y_dyn)
        diffs = treat - base_dyn_resim
        ate, lo, hi = bootstrap_mean_ci(diffs, n_boot=2000, seed=12000 + seed_idx)
        p = paired_permutation_pvalue(diffs, n_perm=3000, seed=13000 + seed_idx)
        rows_ate.append({
            'seed': seed_idx, 'mechanism': 'dynamics', 'intensity': level,
            'ate': ate, 'ci_low': lo, 'ci_high': hi, 'p_value': p,
            'alpha': alpha, 'n_pairs': len(diffs),
        })
        for c, sc in signed_coverage(pred_dyn['mu'], pred_dyn['var'], y_dyn).items():
            rows_signed.append({
                'seed': seed_idx, 'mechanism': 'dynamics', 'intensity': level,
                'level': c, 'signed_cov': sc})

    env.close()
    return rows_null, rows_ate, rows_signed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True, choices=['hopper', 'walker2d'])
    args = ap.parse_args()
    env_name = args.env

    print(f'=== Full pipeline for {env_name} ===')
    print(f'  checkpoints root: {CKPT_ROOTS[env_name]}')
    print(f'  dataset id:       {DATASET_IDS[env_name]}')
    print()

    dataset = minari.load_dataset(DATASET_IDS[env_name], download=False)
    print(f'  episodes: {dataset.total_episodes}, steps: {dataset.total_steps}')
    print()

    all_null, all_ate, all_signed = [], [], []
    for seed_idx in SEEDS:
        print(f'  processing seed {seed_idx}...')
        t0 = time.time()
        n, a, s = process_seed(env_name, dataset, seed_idx)
        all_null.extend(n); all_ate.extend(a); all_signed.extend(s)
        print(f'    done in {time.time()-t0:.1f}s')

    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Null control summary ===
    df_null = pd.DataFrame(all_null)
    df_null.to_csv(out_dir / f'{env_name}_null_control.csv', index=False)
    print()
    print('=== NULL CONTROL (should be ≈ 0) ===')
    print(f'  dynamics (mass=1.0) mean ATE: {df_null["null_dynamics_ate"].mean():+.5f}  '
          f'SD {df_null["null_dynamics_ate"].std():.5f}')
    print(f'  policy   (scale=1.0) mean ATE: {df_null["null_policy_ate"].mean():+.5f}  '
          f'SD {df_null["null_policy_ate"].std():.5f}')

    # === Corrected ATE ===
    df_ate = pd.DataFrame(all_ate)
    df_ate.to_csv(out_dir / f'{env_name}_interventions_corrected.csv', index=False)

    agg = df_ate.groupby(['mechanism', 'intensity']).agg(
        ate_mean=('ate', 'mean'),
        ate_sd=('ate', 'std'),
        alpha_mean=('alpha', 'mean'),
    ).reset_index()

    print()
    print('=== CORRECTED ATE (mean over 5 seeds) ===')
    print(f'{"mechanism":<14} {"intensity":<8} {"ate":>12} {"sd":>10}')
    print('-' * 50)
    for _, r in agg.iterrows():
        print(f"{r['mechanism']:<14} {r['intensity']:<8} "
              f"{r['ate_mean']:>+12.5f} {r['ate_sd']:>10.5f}")

    # === t-CI on corrected ATE ===
    ci_rows = []
    for (mech, level), g in df_ate.groupby(['mechanism', 'intensity']):
        m, lo, hi = t_ci(g['ate'].values)
        ci_rows.append({
            'mechanism': mech, 'intensity': level,
            'mean': m, 'lo': lo, 'hi': hi,
            'excl0': (lo > 0) or (hi < 0),
        })
    df_ci = pd.DataFrame(ci_rows).sort_values(['mechanism', 'intensity']).reset_index(drop=True)
    df_ci.to_csv(out_dir / f'{env_name}_interventions_t_ci.csv', index=False)

    print()
    print('=== t-CI 95% on corrected ATE (n=5) ===')
    print(f'{"condition":<22} {"mean":>10} {"t-CI":>28} {"excl0":>8}')
    print('-' * 72)
    for _, r in df_ci.iterrows():
        ci = f'[{r["lo"]:+.5f}, {r["hi"]:+.5f}]'
        print(f'{r["mechanism"]+"_"+r["intensity"]:<22} '
              f'{r["mean"]:>+10.5f} {ci:>28} {"YES" if r["excl0"] else "NO":>8}')

    # === Recalibration ===
    print()
    print('=== Recalibration (alpha on cal split, nominal 0.90) ===')
    for seed_idx in SEEDS:
        sub = df_ate[df_ate['seed'] == seed_idx]
        alpha = sub['alpha'].iloc[0]
        print(f'  seed {seed_idx}: alpha = {alpha:.4f}')

    # === Signed coverage ===
    df_signed = pd.DataFrame(all_signed)
    df_signed.to_csv(out_dir / f'{env_name}_signed_coverage.csv', index=False)

    print()
    print('=== Signed coverage (mean over 5 seeds) ===')
    summary = df_signed.groupby(['mechanism', 'intensity', 'level'])['signed_cov'].mean().reset_index()
    print(f'{"condition":<22}', end='')
    for c in NOMINAL_LEVELS:
        print(f'{c:>8.2f}', end='')
    print()
    for (mech, level) in sorted(summary[['mechanism','intensity']].drop_duplicates().itertuples(index=False, name=None)):
        print(f'{mech+"_"+level:<22}', end='')
        for c in NOMINAL_LEVELS:
            v = summary[(summary['mechanism']==mech) & (summary['intensity']==level) & (summary['level']==c)]['signed_cov'].values
            print(f'{v[0]:>+8.3f}' if len(v) else f'{"-":>8}', end='')
        print()

    print()
    print(f'saved: {out_dir}/{env_name}_null_control.csv')
    print(f'saved: {out_dir}/{env_name}_interventions_corrected.csv')
    print(f'saved: {out_dir}/{env_name}_interventions_t_ci.csv')
    print(f'saved: {out_dir}/{env_name}_signed_coverage.csv')


if __name__ == '__main__':
    main()