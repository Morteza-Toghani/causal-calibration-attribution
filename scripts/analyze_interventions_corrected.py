"""Corrected interventions analysis: use re-simulated baseline for D and P."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import minari

from src.calibration.instance import instance_calibration_error
from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.interventions.dynamics import dynamics_next_state_from_env
from src.interventions.observation import compute_obs_scale, make_observation_noise
from src.interventions.policy import action_scaler, resimulate_with_new_action
from src.model.ensemble_v2 import load_member, predict_ensemble


PAPER1 = Path.home() / 'paper1_run' / 'experiments' / 'hopper'


def bootstrap_mean_ci(diffs, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(low), float(high)


def paired_perm_pvalue(diffs, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    obs = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)))
    pm = np.abs((signs * d).mean(axis=1))
    return float((1.0 + np.sum(pm >= obs)) / (n_perm + 1.0))


def process_seed(dataset, seed_idx):
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
    base_instance = instance_calibration_error(base_pred['mu'], base_pred['var'], test_y)
    obs_scale = compute_obs_scale(test_obs)
    rng = np.random.default_rng(5000 + seed_idx)
    for level, noise_fraction in (('low', 0.05), ('high', 0.10)):
        eps = make_observation_noise(test_obs.shape, obs_scale, noise_fraction, rng)
        pred_o = predict_ensemble(members, test_obs, test_act, observation_noise=eps)
        treat = instance_calibration_error(pred_o['mu'], pred_o['var'], test_y)
        diffs = treat - base_instance
        ate, lo, hi = bootstrap_mean_ci(diffs, seed=7000 + seed_idx)
        p = paired_perm_pvalue(diffs, seed=8000 + seed_idx)
        rows.append({'seed': seed_idx, 'mechanism': 'observation', 'intensity': level,
                     'ate_original': ate, 'ate_corrected': ate,
                     'ci_low': lo, 'ci_high': hi, 'p_value': p, 'n_pairs': len(diffs),
                     'resim_bias': 0.0})

    env = dataset.recover_environment()
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    base_pred_probe = predict_ensemble(members, probe_obs, probe_act)
    base_inst_orig = instance_calibration_error(
        base_pred_probe['mu'], base_pred_probe['var'], probe_y)
    y_resim_orig = resimulate_with_new_action(env, probe_obs, probe_act)
    base_inst_corr = instance_calibration_error(
        base_pred_probe['mu'], base_pred_probe['var'], y_resim_orig)
    resim_bias_policy = float((base_inst_corr - base_inst_orig).mean())

    for level, scale in (('low', 0.95), ('high', 0.80)):
        transform = action_scaler(scale, action_low, action_high)
        new_act = transform(probe_act)
        y_resim_new = resimulate_with_new_action(env, probe_obs, new_act)
        pred_p = predict_ensemble(members, probe_obs, new_act)
        treat = instance_calibration_error(pred_p['mu'], pred_p['var'], y_resim_new)

        ate_orig = float((treat - base_inst_orig).mean())
        diffs_corr = treat - base_inst_corr
        ate_corr, lo, hi = bootstrap_mean_ci(diffs_corr, seed=9000 + seed_idx)
        p = paired_perm_pvalue(diffs_corr, seed=10000 + seed_idx)

        rows.append({'seed': seed_idx, 'mechanism': 'policy', 'intensity': level,
                     'ate_original': ate_orig, 'ate_corrected': ate_corr,
                     'ci_low': lo, 'ci_high': hi, 'p_value': p,
                     'n_pairs': len(diffs_corr), 'resim_bias': resim_bias_policy})

    dyn_n = min(len(test_obs), 5000)
    rng_dyn = np.random.default_rng(11000 + seed_idx)
    dyn_idx = np.sort(rng_dyn.choice(len(test_obs), dyn_n, replace=False))
    dyn_obs, dyn_act = test_obs[dyn_idx], test_act[dyn_idx]
    dyn_y_dataset = test_y[dyn_idx]

    pred_d = predict_ensemble(members, dyn_obs, dyn_act)

    base_inst_orig = instance_calibration_error(pred_d['mu'], pred_d['var'], dyn_y_dataset)
    y_resim_1, _ = dynamics_next_state_from_env(
        env, dyn_obs, dyn_act, mass_scale=1.0, seed=123 + seed_idx)
    base_inst_corr = instance_calibration_error(pred_d['mu'], pred_d['var'], y_resim_1)
    resim_bias_dyn = float((base_inst_corr - base_inst_orig).mean())

    for level, mass_scale in (('low', 0.95), ('high', 0.80)):
        y_resim_treat, _ = dynamics_next_state_from_env(
            env, dyn_obs, dyn_act, mass_scale=mass_scale, seed=123 + seed_idx)
        treat = instance_calibration_error(pred_d['mu'], pred_d['var'], y_resim_treat)

        ate_orig = float((treat - base_inst_orig).mean())
        diffs_corr = treat - base_inst_corr
        ate_corr, lo, hi = bootstrap_mean_ci(diffs_corr, n_boot=2000, seed=12000 + seed_idx)
        p = paired_perm_pvalue(diffs_corr, n_perm=3000, seed=13000 + seed_idx)

        rows.append({'seed': seed_idx, 'mechanism': 'dynamics', 'intensity': level,
                     'ate_original': ate_orig, 'ate_corrected': ate_corr,
                     'ci_low': lo, 'ci_high': hi, 'p_value': p,
                     'n_pairs': len(diffs_corr), 'resim_bias': resim_bias_dyn})

    env.close()
    return rows


def main():
    dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=False)

    all_rows = []
    for seed_idx in range(5):
        print(f'processing seed {seed_idx}...')
        all_rows.extend(process_seed(dataset, seed_idx))

    df = pd.DataFrame(all_rows)
    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'interventions_corrected.csv', index=False)
    print(f'saved {out_dir / "interventions_corrected.csv"}')

    agg = df.groupby(['mechanism', 'intensity']).agg(
        ate_orig_mean=('ate_original', 'mean'),
        ate_corr_mean=('ate_corrected', 'mean'),
        ate_corr_sd=('ate_corrected', 'std'),
        bias=('resim_bias', 'mean'),
    ).reset_index()

    md = ['# Interventions - original vs corrected', '']
    md.append('Correction: for D and P, baseline target is now y_resim(scale=1.0)')
    md.append('instead of y_dataset. This removes the re-simulation bias.')
    md.append('')
    md.append('| mechanism | intensity | ATE (original) | ATE (corrected) | SD | re-sim bias |')
    md.append('|---|---|---|---|---|---|')
    for _, r in agg.iterrows():
        md.append(f"| {r['mechanism']} | {r['intensity']} | "
                  f"{r['ate_orig_mean']:+.5f} | {r['ate_corr_mean']:+.5f} | "
                  f"{r['ate_corr_sd']:.5f} | {r['bias']:+.5f} |")

    with open(out_dir / 'interventions_comparison.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'saved {out_dir / "interventions_comparison.md"}')

    print()
    print('=== Original vs corrected ATE (mean over 5 seeds) ===')
    print(f'{"mechanism":<14} {"intensity":<8} {"orig":>12} {"corrected":>12} {"bias":>10}')
    print('-' * 60)
    for _, r in agg.iterrows():
        print(f"{r['mechanism']:<14} {r['intensity']:<8} "
              f"{r['ate_orig_mean']:>12.5f} {r['ate_corr_mean']:>12.5f} {r['bias']:>+10.5f}")


if __name__ == '__main__':
    main()